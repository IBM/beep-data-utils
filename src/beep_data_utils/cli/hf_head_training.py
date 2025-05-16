"""This module implements the training of the hf-compatible heads
"""

import json
import math
from os import path
from typing import Dict, List, Tuple

import click
import mlflow
import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from ..modeling.heads.hf_utils import SeqDataset
from ..modeling.heads.registries import HF_HEAD, HF_HEAD_CONFIG


@click.command()
@click.option(
    "-d",
    "--data-dir",
    help="The directory where the data will be, expected files are train.csv, test.csv, eval.csv OR you can load the independently using the --data-train=path/to/file.csv options",
    required=False,
    type=click.Path(exists=True),
    show_default=True,
)
@click.option(
    "-train",
    "--data-train-path",
    help="Path to the train dataset file, not a directory",
    default="data/train.csv",
    required=False,
    type=click.Path(exists=False),
    show_default=True,
)
@click.option(
    "-test",
    "--data-test-path",
    default="data/test.csv",
    help="Path to the test dataset file, not a directory",
    required=False,
    type=click.Path(exists=False),
    show_default=True,
)
@click.option(
    "-eval",
    "--data-eval-path",
    default="data/eval.csv",
    help="Path to the evaluation dataset file, not a directory",
    required=False,
    type=click.Path(exists=False),
    show_default=True,
)
@click.option(
    "-m",
    "--head-type",
    "--model-type",
    default="LSTM",
    help=f"The type of head you want to train MUST be one of {','.join(HF_HEAD.keys())}",
    type=str,
    required=False,
)
@click.option(
    "-c",
    "--conf",
    "--model-config",
    help="Path where the model configuration is stored see HF model for more info, can also give a the config as a json string",
    required=True,
    type=str,
)
@click.option(
    "-a",
    "--args",
    "--training-arguments",
    help="Path where the training argument is stored (json) format see the HF trainingArgument class documentation, for the possible key-value. These argument are related ONLY to the training loops (optimizer, batch ...)",
    required=True,
    type=click.Path(exists=False),
)
@click.option(
    "-i",
    "--input-features",
    help="A comma separated list features name to use for prediction",
    type=str,
    required=False,
)
@click.option(
    "--targets",
    help="A comma separated list of target name for prediction. These value will be predicted",
    default="charge_capacity",
    type=str,
    required=False,
)
@click.option(
    "-u",
    "--time_unit",
    help="The name of the column containing timestamp for your timeseries",
    default="cycle_number",
    type=str,
    required=False,
)
@click.option(
    "-id",
    "--col_id",
    help="The name of the column containing id of your devices",
    default="battery",
    type=str,
    required=False,
)
@click.option(
    "--seq_len",
    default=100,
    help="How many elements are in the sequence",
)
@click.option(
    "--output_dir",
    help="Using this option you can overwrite the output_dir defined in the training args file",
    required=False,
    type=str,
)
@click.option(
    "--normalize",
    help="Using this option you can normalize the input values if it hasn't been done yet",
    required=False,
    is_flag=True,
    default=False,
)
@click.option(
    "--smooth_charge_cap",
    help="Using this option you can smooth the charge capacity if this is your target",
    required=False,
    is_flag=True,
    default=False,
)
@click.option(
    "--do_split",
    help="Using this option you can merge all the data and create a per-device split",
    required=False,
    is_flag=True,
    default=True,
)
def main(
    data_dir,
    data_train_path,
    data_test_path,
    data_eval_path,
    head_type,
    conf,
    args,
    input_features,
    targets,
    time_unit,
    col_id,
    seq_len,
    output_dir,
    normalize,
    smooth_charge_cap,
    do_split,
):
    """ """
    logger.debug(
        f"Running with the following arguments:\n{data_dir=},{data_train_path=},{data_test_path=},{data_eval_path=},{head_type=},{conf=},{args=},{input_features=},{targets=},{time_unit=},{col_id=},{seq_len=},{output_dir=},{normalize=},{smooth_charge_cap=},"
    )
    assert (
        head_type in HF_HEAD
    ), f"{head_type} is not a valid model head, must be one of {', '.join(HF_HEAD.keys())}"
    assert not (
        data_dir is None and data_train_path is None
    ), "You need to either give a path to the directory with the files (train.csv, test.csv, eval.csv) in it OR a path to each file"

    # The file is not given
    if data_dir is not None:
        data_train_path = path.join(data_dir, "train.csv")
        data_test_path = path.join(data_dir, "test.csv")
        data_eval_path = path.join(data_dir, "eval.csv")

    train_df = read_csv_to_pl(data_train_path)
    test_df = read_csv_to_pl(data_test_path)
    eval_df = read_csv_to_pl(data_eval_path)
    assert (
        set(train_df.columns) == set(eval_df.columns)
    ), "Train and eval sets are not part of the same dataset, because their columns are not identical"
    if do_split:
        logger.info("Doing our own train-test-eval split on a device level")
        train_df, test_df, eval_df = train_test_eval_split(
            train_df, test_df, eval_df, col_id=col_id
        )
        logger.info(f"{train_df.shape=},{test_df.shape=},{eval_df.shape=}")
    TARGET_FEATURES, INPUT_FEATURES = [], []
    if targets is not None:
        TARGET_FEATURES = targets.split(",")
    if input_features is not None and input_features != "":
        INPUT_FEATURES = input_features.split(",")
    else:
        INPUT_FEATURES = infer_feature_list(train_df, TARGET_FEATURES + [col_id, time_unit])
        logger.info("Input features infered")

    logger.info("Running with the following data configuration:")
    logger.info(f"Target features = {', '.join(TARGET_FEATURES)}")
    logger.info(f"Input features = {', '.join(INPUT_FEATURES)}")

    model, training_args = load_configs(
        args_path=args,
        head_type=head_type,
        conf=conf,
        default_input_feature_len=len(INPUT_FEATURES) + 1,
        default_output_feature_len=len(TARGET_FEATURES),
        default_seq_len=seq_len,
    )
    logger.info(f"Training args:\n{training_args}")
    if output_dir is not None and output_dir != "":
        logger.info(f"Overwriting output_dir to {output_dir}")
        training_args.output_dir = output_dir
    assert (
        model.config.num_input_feature == len(INPUT_FEATURES) + 1
    ), f"Your model configuration and your dataset configuration do not match {model.config.num_input_feature=} != {len(INPUT_FEATURES) + 1=}"
    logger.info(f"{head_type} head initialized...")
    logger.info(f"Model config: \n{model.config}")
    logger.debug("Training")
    train_dataset = SeqDataset(
        dataset=train_df,
        input_features=INPUT_FEATURES,
        target_features=TARGET_FEATURES,
        cutoff=seq_len,
        normalize_columns=normalize,
        monoton_smooth=["charge_capacity"] if smooth_charge_cap else [],
    )
    logger.debug("Eval")
    eval_dataset = SeqDataset(
        dataset=eval_df,
        input_features=INPUT_FEATURES,
        target_features=TARGET_FEATURES,
        cutoff=seq_len,
        normalize_columns=normalize,
        monoton_smooth=["charge_capacity"] if smooth_charge_cap else [],
    )
    logger.debug("Test")
    test_dataset = SeqDataset(
        dataset=test_df,
        input_features=INPUT_FEATURES,
        target_features=TARGET_FEATURES,
        cutoff=seq_len,
        normalize_columns=normalize,
        monoton_smooth=["charge_capacity"] if smooth_charge_cap else [],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    logger.info("Training done")
    logger.info(mlflow.active_run())
    with mlflow.start_run():
        for split_dataset, split_name in zip(
            [train_dataset, test_dataset, eval_dataset], ["train", "test", "eval"]
        ):
            logger.info(f"Running prediction for {split_name}")
            pred_y = trainer.predict(split_dataset, metric_key_prefix=split_name)
            for metric_name, metric_value in pred_y.metrics.items():
                mlflow.log_metric(key=f"best_model_{split_name}_{metric_name}", value=metric_value)
            assert pred_y.predictions.shape[0] == pred_y.label_ids.shape[0]
            r = np.concatenate([pred_y.label_ids, pred_y.predictions], axis=1)
            pl.DataFrame(
                r,
                schema=[f"true_{x}" for x in TARGET_FEATURES]
                + [f"predicted_{x}" for x in TARGET_FEATURES],
            ).write_csv(f"{output_dir}/model_prediction_{split_name}.csv")


def read_csv_to_pl(path: str):
    """Read the csv path to a polars dataframe.
    CSV is expected to have a header and a column index having battery_name/time
    Args:
        path: The path to a csv file to load
    Returns:
        Return a new polars DataFrame containing the content from the csv file."""
    logger.debug(f"Reading {path}")
    return pl.read_csv(path, has_header=True).with_columns(
        pl.col("index").str.split(by="/").list.first().alias("battery")
    )


def infer_feature_list(df: pl.DataFrame, exclude_list: List[str] = []) -> List[str]:
    """Extract numeric columns name to be used a input feature

    Args:
        df (pl.DataFrame): The dataframe on which we extract the columns name
        exclude_list (list, optional): a list of feature name you want to discard, should be set to any output feature. Defaults to [].

    Returns:
        List[str]: The names of the columns who are numeric and not part of the excluded list.
    """
    return [c for c in df.columns if (df.schema[c].is_numeric() and c not in exclude_list)]


def load_configs(
    args_path: str,
    head_type: str,
    conf: str,
    default_input_feature_len: int = -1,
    default_output_feature_len=-1,
    default_seq_len=-1,
) -> Tuple[PreTrainedModel, TrainingArguments]:
    """Loads the following config for training: ModelConfig, based on the Model type and the training arguments for the HF-trainer

    Args:
        args_path (str): a json or yaml file containing the info to populate an instance of TrainingArguments
        head_type (str): The type of head we aim to train, see the HF_HEAD for a list
        conf (str): The configuration to initilize the model, can be a json string or a path to a yaml / json file

    Returns:
        Tuple[PreTrainedModel,TrainingArguments]: return an initialized model and the training arguments
    """
    parser = HfArgumentParser(TrainingArguments)
    training_args = (None,)
    if args_path.endswith(".json"):
        (training_args,) = parser.parse_json_file(json_file=args_path)
    elif args_path.endswith(".yaml") or args_path.endswith(".yml"):
        (training_args,) = parser.parse_yaml_file(yaml_file=args_path)

    if conf.endswith("json"):
        model_config = HF_HEAD_CONFIG[head_type].from_pretrained(conf)
    elif conf != "":
        dict_conf = json.loads(conf)
        model_config = HF_HEAD_CONFIG[head_type](**dict_conf)
    else:
        model_config = HF_HEAD_CONFIG[head_type](
            num_input_feature=default_input_feature_len,
            num_output_feature=default_output_feature_len,
        )
        if head_type == "linear":
            model_config.seq_len = default_seq_len
    model = HF_HEAD[head_type](model_config)
    return model, training_args


def compute_metrics(eval_pred: EvalPrediction, compute_result: bool = True) -> Dict[str, float]:
    """Compute the MSE, R2, RMSE and MAE metrics for the sample.
    Args:
        eval_pred (EvalPrediction): Contains at least predictions and label_ids to compute a metrics
        compute_result (bool): Unused, for HF API compatibility

    Returns:
        Dict[Str,float]: return the different score computed, keys are `mse`,`r2`,`rmse`,`mae`

    """
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if len(predictions.shape) > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    if len(label_ids.shape) > 1 and label_ids.shape[1] == 1:
        label_ids = label_ids.flatten()
    return {
        "mse": mean_squared_error(label_ids, predictions),
        "r2": r2_score(label_ids, predictions),
        "rmse": root_mean_squared_error(label_ids, predictions),
        "mae": mean_absolute_error(label_ids, predictions),
    }


def train_test_eval_split(
    train_df: pl.DataFrame, test_df: pl.DataFrame, eval_df: pl.DataFrame, col_id: str = "battery"
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Generate a new train test split based on the device id. Each split has no overlapping devices.
    Args:
        train_df (pl.DataFrame): This is the train dataset. It can be a with overlapping devices from the other sets.
        test_df  (pl.DataFrame): This is the test dataset. It can be a with overlapping devices from the other sets.
        eval_df  (pl.DataFrame): This is the eval dataset. It can be a with overlapping devices from the other sets.
        col_id (str): This indicate the column name containig the battery id.
    Returns:
        Return a new split with non overlapping device
    """
    data = pl.concat([train_df, test_df, eval_df], how="vertical")
    device_list = data.get_column(col_id).unique().shuffle(seed=40).to_list()
    ratio = {"train": 0.7, "test": 0.15, "validation": 0.15}

    # Calculate number of devices for each split
    num_devices = len(device_list)
    train_size = int(math.floor(num_devices * ratio["train"]))
    test_size = int(math.ceil(num_devices * ratio["test"]))

    train_devices = device_list[:train_size]
    test_devices = device_list[train_size : train_size + test_size]
    validation_devices = device_list[train_size + test_size :]
    assert (
        len(validation_devices) > 0
    ), "You have an empty validation set, please change the ratio used for the split, do you own split by setting do_split = False or get a bigger dataset"
    assert (
        len(test_devices) > 0
    ), "You have an empty test set, please change the ratio used for the split, do you own split by setting do_split = False or get a bigger dataset"
    # Create dataframes for each split based on device IDs
    train = data.filter(pl.col(col_id).is_in(train_devices))
    test = data.filter(pl.col(col_id).is_in(test_devices))
    validation = data.filter(pl.col(col_id).is_in(validation_devices))
    return train, test, validation


# Allows for module debugging
if __name__ == "__main__":
    main()
