"""This module allows the training of a random forest (RF) on a list of embeddings, additionally, splitting is done per cycle"""

import ast
import json
import math
from os import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    KFold,
    RepeatedKFold,
    ShuffleSplit,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from ..configuration import MODEL_SETTINGS

CV_TYPES = {
    "kfold": KFold,
    "repeated_kfold": RepeatedKFold,
}


def cum_max_smooth(data: pd.DataFrame, time_unit, col_to_smooth, column_id, ceiling_value=1.1):
    dfs_pandas = []
    for device in data[column_id].unique():
        # make sure that you have a sorted dataset (per cycle_number decreasing)
        df_device = data[data[column_id] == device].copy()
        df_device[col_to_smooth] = df_device[col_to_smooth].apply(
            lambda x: ceiling_value if x > ceiling_value else x
        )
        df_device[col_to_smooth] = df_device[col_to_smooth].cummax()
        dfs_pandas.append(df_device)
    df = pd.concat(dfs_pandas)
    return df


def process_dataset(
    data: pd.DataFrame,
    input_feature_names: List[str] = [],
    target_feature_names: List[str] = [],
    agg_fn: str = "max",
    cycle_number_cutoff: int = 100,
    column_id: str = "battery",
    time_unit: str = "cycle_number",
    smooth: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This function prepare the dataset (multivariates time-series) for a supervised training.
    The data is split into two parts X the input of your ML model and y the target to be predicted.


    Args:
        data (pd.DataFrame): The data to process, the expected format is a list of items, each items having multiple sensors each having multiple recordings per cycle.
        the index is battery_name/time
        cycle_cutoff (int): the number of cycle we should consider for training. We usually take the first 100 cycles and predict the next ones
        target_names (List[str]): The list of feature to use as target
        agg_fn (str): A function aggregating value per cycle. Usually you'd use min,max or mean.
        scaling_fn (Optional[Callable[[pd.DataFrame],pd.DataFrame]]) If not None, apply an optional scaling to all the features except the targets
        column_id (str) The column name containing the devices ID
        time_unit (str) The column name containing the timestamps

    Returns:
        tuple: processed_data, targets, battery_ids
    """
    assert time_unit in data.columns, f"{time_unit} is not part of the data"
    assert column_id in data.columns, f"{column_id} is not part of the data"
    data[column_id] = data.index.str.split("/").str[0]
    data = data.reset_index(drop=True)
    if smooth:
        data = data.sort_values(by=[column_id, time_unit], ascending=False)
        data = cum_max_smooth(
            data,
            time_unit=time_unit,
            col_to_smooth=target_feature_names,
            column_id=column_id,
            ceiling_value=1.1,
        )
    training_data = data[data[time_unit] <= cycle_number_cutoff]
    target_data = data[data[time_unit] > cycle_number_cutoff]
    training_data = training_data[input_feature_names + [column_id]].groupby(column_id).agg("mean")
    target_data = target_data[target_feature_names + [column_id, time_unit]]
    target_data = target_data.groupby([column_id, time_unit])[target_feature_names].agg(agg_fn)
    # put the multi index (column_id, time_unit) back into the columns
    target_data = target_data.reset_index()
    r = pd.merge(training_data, target_data, how="inner", on=column_id, suffixes=("_l", "_r"))
    assert (
        "_l" not in r.columns and "_r" not in r.columns
    ), f" You got overlapping columns: {set([x for x in r.columns if '_l' in x or '_r' in x])}"
    # the length is input+output features +2 because we have the time_unit and the col_id in the result
    assert (
        r.shape[1] == (len(target_feature_names) + len(input_feature_names) + 2)
    ), f"Got {r.shape[1]=}, expected {len(input_feature_names) + len(target_feature_names)},\n got:{(len(r.columns)), sorted(r.columns)}\n Wanted:{len(sorted(set(input_feature_names + target_feature_names))), sorted(set(input_feature_names + target_feature_names))}"
    X = r.loc[:, input_feature_names + [time_unit]]
    X.rename({time_unit: f"target_{time_unit}"}, inplace=True)
    y = r.loc[:, target_feature_names]
    return (
        X,
        y,
        r.loc[:, column_id],
    )


def generate_cross_validation_pipeline(
    regressor: RegressorMixin,
    parameters_grid: Dict[str, Any],
    cv: BaseCrossValidator = RepeatedKFold(n_splits=5, n_repeats=1, random_state=12345),
    feature_selector: Optional[SelectorMixin] = VarianceThreshold(),
    feature_transformer: Optional[TransformerMixin] = MinMaxScaler(),
    random_state: int = 12345,
    number_of_jobs: int = 1,
    scoring: Any = None,
    refit: Any = True,
):
    """Evaluate a regressor via cross validation.

    Args:
        regressor: a regressor.
        parameters_grid: parameters grid.
        cv: cross validator.
        folds: number of cross validation folds,
            defaults to 5.
        repeats: number of cross validation repeats,
            defaults to 1.
        random_state: random state, defaults to 12345.
        number_of_jobs: number of jobs to run in parallel. Defaults to 1.
            -1 means using all processors.
        scoring: scoring function or functions to evaluate predictions on the test set.
            Defaults to None to use the regressor default score method.
        refit: whether to refit with best estimator, can be str, boolean or callable.
            For multiple metric evaluation, this needs to be a string denoting the scorer
            is used to find the best parameters for refitting the estimator at the end.

    Returns:
        an evaluation report.
    """
    # ensure reproducibility in the classifier and log seed via parameter
    steps = []
    if feature_selector is not None:
        steps.append(feature_selector)
    if feature_transformer is not None:
        steps.append(feature_transformer)
    # NOTE: this condition checks whether we have more than one parameter set to optimize via CV
    if sum(len(parameter_values) for parameter_values in parameters_grid.values()) > len(
        parameters_grid
    ):
        cv_splitter = cv
    else:
        # NOTE: in this case CV is not needed (no parameters to optimize), hence we only do a single split
        # using as many data as possible (5% for validation is an arbitrary low number)
        cv_splitter = ShuffleSplit(
            n_splits=1,
            test_size=MODEL_SETTINGS.single_split_validation_size,
            random_state=random_state,
        )
    # generate the pipeline
    return make_pipeline(
        *steps,
        GridSearchCV(
            regressor,
            param_grid=parameters_grid,
            cv=cv_splitter,
            refit=refit,
            n_jobs=number_of_jobs,
            scoring=scoring,
            return_train_score=True,
        ),
    )


@click.command()
@click.option(
    "--train_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to training data CSV",
)
@click.option(
    "--val_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to validation data CSV",
)
@click.option(
    "--test_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to test data CSV",
)
@click.option(
    "--output_path", required=True, type=click.Path(path_type=Path), help="Path to save results"
)
@click.option(
    "--target_name",
    required=True,
    default="charge_capacity",
    type=str,
    help="Names of target columns, comma separated",
)
@click.option(
    "--features_name",
    required=True,
    default="val_0,val_1,val_2,val_3,val_4,val_5,val_6,val_7,val_8,val_9,val_10,val_11,val_12,val_13,val_14,val_15,val_16,val_17,val_18,val_19,val_20,val_21,val_22,val_23,val_24,val_25,val_26,val_27,val_28,val_29,val_30,val_31,val_32,val_33,val_34,val_35,val_36,val_37,val_38,val_39,val_40,val_41,val_42,val_43,val_44,val_45,val_46,val_47,charge_policy_Q1,charge_policy_Q2,charge_policy_Q3,charge_policy_Q4",
    type=str,
    help="Names of feature columns, comma separated",
)
@click.option(
    "--col_id",
    required=True,
    default="battery",
    type=str,
    help="Name of the columns containing the device id",
)
@click.option(
    "--time_unit",
    required=True,
    default="cycle_number",
    type=str,
    help="Names of the columns containing the timestamps",
)
@click.option(
    "--cycle_cutoff",
    required=True,
    type=int,
    default=100,
    help="Number of cycles to use for training",
)
@click.option(
    "--cv_type",
    type=click.Choice(list(CV_TYPES.keys())),
    default="kfold",
    help="Type of cross validation to use",
)
@click.option("--n_splits", default=5, type=int, help="Number of splits for cross validation")
@click.option(
    "--model_parameters_grid",
    required=False,
    type=str,
    help="Model parameters grid for grid search as JSON-formatted string",
)
@click.option(
    "--dataset_metadata_path",
    required=False,
    type=click.Path(path_type=Path, exists=True),
    help="Path to dataset metadata for rescaling",
)
@click.option("--random_state", default=42, type=int, help="Random seed")
@click.option(
    "--agg_fn",
    default="max",
    type=str,
    help="The aggregation function (numpy style) that should by used when aggregating the cycles",
)
@click.option(
    "--smooth_target",
    type=bool,
    help="Smooth the target columns using a cumulative smooth, this works only for a monotonic metrics like charge_capacity which can only decrease",
)
def train_head(
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    output_path: Path,
    target_name: str,
    features_name: str,
    col_id: str,
    time_unit: str,
    cycle_cutoff: int,
    cv_type: str,
    n_splits: int,
    model_parameters_grid: str,
    dataset_metadata_path: Path,
    random_state: int,
    agg_fn: str,
    smooth_target: bool,
):
    features_name_list = features_name.split(",")
    target_name_list = target_name.split(",")
    logger.info(
        f"Using the following:train_data_path\n{val_data_path=}\n{test_data_path=}\n{output_path=}\n{target_name=}\n{features_name_list=}\n{features_name=}\n{col_id=}\n{time_unit=}\n{cycle_cutoff=}\n{cv_type=}\n{n_splits=}\n{model_parameters_grid=}\n{dataset_metadata_path=}\n{random_state=}\n{agg_fn=}\n{smooth_target=}"
    )
    train_data = pd.read_csv(train_data_path, index_col=0)
    val_data = pd.read_csv(val_data_path, index_col=0)
    test_data = pd.read_csv(test_data_path, index_col=0)
    train_data, test_data, val_data = train_test_val_split_per_device(
        train_data, val_data, test_data, col_id=col_id
    )
    assert train_data[train_data[time_unit] > 10].shape[0] != 0, "Please unscale your features"

    logger.debug(f"Processing datasets using first {cycle_cutoff} cycles...")
    logger.debug(f"{train_data.shape = }")
    logger.debug(f"{test_data.shape = }")
    logger.debug(f"{val_data.shape = }")

    X_train, y_train, _ = process_dataset(
        train_data,
        input_feature_names=features_name_list,
        target_feature_names=target_name_list,
        agg_fn=agg_fn,
        cycle_number_cutoff=100,
        column_id=col_id,
        time_unit=time_unit,
        smooth=smooth_target,
    )
    logger.info(f"Train dataset processed: {X_train.shape=}, {y_train.shape=}")
    X_test, y_test, test_batterie_ids = process_dataset(
        test_data,
        input_feature_names=features_name_list,
        target_feature_names=target_name_list,
        agg_fn=agg_fn,
        cycle_number_cutoff=100,
        column_id=col_id,
        time_unit=time_unit,
        smooth=smooth_target,
    )
    logger.info(f"Max charge capacity in test (true): {y_test.max()}")
    logger.info(f"Test dataset processed: {X_test.shape=}, {y_test.shape=}")
    X_val, y_val, val_batterie_ids = process_dataset(
        val_data,
        input_feature_names=features_name_list,
        target_feature_names=target_name_list,
        agg_fn=agg_fn,
        cycle_number_cutoff=100,
        column_id=col_id,
        time_unit=time_unit,
        smooth=smooth_target,
    )
    logger.info(f"Val dataset processed: {X_val.shape=}, {y_val.shape=}")
    targets_name = y_train.columns
    # Delete a warning where the target_data (y_x) is 2D but with a 1-channel prediction.
    if y_val.shape[1] == 1:
        logger.info("Flattening the 1D target array")
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        y_val = y_val.values.ravel()

    logger.info(f"Types of data: {type(X_val)=}, {type(y_val)=}")
    logger.info(f"Types of data: {type(X_test)=}, {type(y_test)=}")
    logger.info(f"Types of data: {type(X_train)=}, {type(y_train)=}")

    parsed_parameters_grid = ast.literal_eval(model_parameters_grid)
    cv = CV_TYPES[cv_type](n_splits=n_splits, shuffle=True, random_state=random_state)
    model = RandomForestRegressor(random_state=random_state)
    pipeline = generate_cross_validation_pipeline(
        regressor=model,
        parameters_grid=parsed_parameters_grid,
        cv=cv,
        feature_selector=None,
        feature_transformer=None,
        random_state=random_state,
        number_of_jobs=-1,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        },
        refit="mae",
    )
    logger.info("Cross validation pipeline generate...")
    logger.info("Training model with cross-validation...")
    pipeline.fit(X_train, y_train)

    cv_results = pd.DataFrame(pipeline[-1].cv_results_)
    cv_results.to_csv(path.join(output_path, "cv_results.csv"))

    logger.info("making prediction on test and val:")
    pred_val = pipeline.predict(X_val)
    pred_test = pipeline.predict(X_test)

    logger.info(f"Validation lengths - Targets: {len(y_val)}, Predictions: {len(pred_val)}")
    logger.info(f"Test lengths - Targets: {len(y_test)}, Predictions: {len(pred_test)}")

    val_metrics = {
        "val_mae": mean_absolute_error(y_val, pred_val),
        "val_mse": mean_squared_error(y_val, pred_val),
        "val_rmse": np.sqrt(mean_squared_error(y_val, pred_val)),  # Fixed
        "val_r2": r2_score(y_val, pred_val),
    }

    test_metrics = {
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_mse": mean_squared_error(y_test, pred_test),
        "test_rmse": np.sqrt(mean_squared_error(y_test, pred_test)),  # Fixed
        "test_r2": r2_score(y_test, pred_test),
    }

    cv_metrics = {}
    for metric in ["mae", "mse", "r2"]:
        cv_metrics[f"cv_{metric}_mean"] = abs(cv_results[f"mean_test_{metric}"].mean())
        cv_metrics[f"cv_{metric}_std"] = cv_results[f"std_test_{metric}"].std()

    metrics = {**cv_metrics, **val_metrics, **test_metrics}

    logger.info("Cross-validation Metrics:")
    for metric, value in cv_metrics.items():
        logger.info(f"{metric}: {value:.8f}")

    logger.info("Validation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"{metric}: {value:.8f}")

    logger.info("Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.8f}")

    output_path.mkdir(parents=True, exist_ok=True)
    with open(path.join(output_path, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.debug(f"{val_batterie_ids.shape = }")
    logger.debug(f"{target_name = }")
    logger.debug(f"{y_val.shape = }")
    logger.debug(f"{pred_val.shape = }")
    logger.info("Creating the resulting dataframe...")

    pred_val = pred_val.reshape(-1, len(targets_name))
    y_val = y_val.reshape(-1, len(targets_name))
    pred_test = pred_test.reshape(-1, len(targets_name))
    y_test = y_test.reshape(-1, len(targets_name))
    val_results_df = pd.DataFrame(
        {
            "battery": val_batterie_ids,
            "dataset": "validation",
        }
        # Append to the above dictionary the target columns with their *ground-truth*
        | {c: y_val[:, i] for i, c in enumerate(targets_name)}
        # Append to the dictionary the target columns with their *predicted* value
        # as pred_val is a numpy array we slice it per columns
        | {f"predicted_{c}": pred_val[:, i] for i, c in enumerate(targets_name)}
    )

    test_results_df = pd.DataFrame(
        {
            "battery": test_batterie_ids,
            "dataset": "test",
        }
        # Append to the above dictionary the target columns with their *ground-truth*
        | {c: y_test[:, i] for i, c in enumerate(targets_name)}
        # Append to the dictionary the target columns with their *predicted* value
        | {f"predicted_{c}": pred_test[:, i] for i, c in enumerate(targets_name)}
    )
    pd.concat([val_results_df, test_results_df]).to_csv(path.join(output_path, "predictions.csv"))
    logger.debug(f"columns, {val_results_df.columns}, {test_results_df.columns}")
    logger.debug(f"shapes are, {val_results_df.shape}, {test_results_df.shape}")
    logger.debug("RF trained and evaluated")


def train_test_val_split_per_device(train, test, val, col_id="battery"):
    df = pd.concat([train, test, val])
    df[col_id] = df.index.str.split("/").str[0]
    device_list = df[col_id].unique().tolist()
    ratio = {"train": 0.7, "test": 0.15, "validation": 0.15}

    # Calculate number of devices for each split
    num_devices = len(device_list)
    train_size = int(math.floor(num_devices * ratio["train"]))
    test_size = int(math.ceil(num_devices * ratio["test"]))

    train_devices = set(device_list[:train_size])
    test_devices = set(device_list[train_size : train_size + test_size])
    val_devices = set(device_list[train_size + test_size :])
    df = df.sample(frac=1)
    train = df[df[col_id].isin(train_devices)]
    test = df[df[col_id].isin(test_devices)]
    val = df[df[col_id].isin(val_devices)]
    return train, test, val


if __name__ == "__main__":
    train_head()
