"""Fine-tuning and evaluation with cycle-based splitting."""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, RepeatedKFold

from ..configuration import MODEL_SETTINGS
from .time_enforcing_rf_training import generate_cross_validation_pipeline

DEFAULT_RF_PARAMS = {
    "max_depth": [8],
    "n_estimators": [200],
    "min_samples_split": [10],
    "min_samples_leaf": [8],
    "max_features": ["sqrt"],
    "bootstrap": [True],
    "max_samples": [0.9],
    "criterion": ["squared_error"],
    "random_state": [42],
}


def get_metrics(predictions: np.ndarray, targets: np.ndarray, batteries: list) -> Dict[str, float]:
    """Calculate metrics."""
    window_level_mae = mean_absolute_error(targets, predictions)

    # Average predictions per battery
    df = pd.DataFrame({"battery": batteries, "prediction": predictions, "target": targets})
    battery_averages = df.groupby("battery").agg({"prediction": "mean", "target": "first"})
    pred_avg, targets_avg = battery_averages["prediction"].values, battery_averages["target"].values

    metrics = {
        "mae": mean_absolute_error(targets_avg, pred_avg),
        "mape": mean_absolute_percentage_error(targets_avg, pred_avg) * 100,  # to get percentage
        "mse": mean_squared_error(targets_avg, pred_avg),
        "rmse": np.sqrt(mean_squared_error(targets_avg, pred_avg)),
        "r2": r2_score(targets_avg, pred_avg),
        "window_level_mae": window_level_mae,
    }

    return metrics


def process_dataset(
    data: pd.DataFrame,
    cycle_cutoff: int,
    target_name: str,
    avg_cycles: bool = False,
    cycle_window: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """Process dataset by filtering to early cycles and averaging cycle windows"""
    excluded_prefixes = ["target", "index", "cycle", "time"]
    features = [
        col
        for col in data.columns
        if not any(col.startswith(prefix) for prefix in excluded_prefixes)
    ]

    data = data.copy()
    data["battery"] = data.index.str.split("/").str[0]
    # data = data[~data["battery"].str.startswith("ncmca")]

    # data = data[data["target_cycle_life"] > 100]  # Needed for NCA NCM dataset
    valid_batteries = data.groupby("battery")[target_name].first()
    valid_batteries = valid_batteries[
        valid_batteries != MODEL_SETTINGS.unknown_numerical_value
    ].index
    data = data[data["battery"].isin(valid_batteries)]

    early_cycles_data = data[data["cycle_number"] <= cycle_cutoff]

    if avg_cycles:
        early_cycles_data["cycle_window"] = (early_cycles_data["cycle_number"] - 1) // cycle_window

        # Average features within each window for each battery
        aggregated_data = (
            early_cycles_data.groupby(["battery", "cycle_window"])
            .agg({
                **{feature: "mean" for feature in features},
                target_name: "first",
            })
            .reset_index()
        )
        processed_features = aggregated_data[features]
        targets = aggregated_data[target_name].values
        batteries = aggregated_data["battery"].tolist()
    else:
        # No averaging, use data as is
        processed_features = early_cycles_data[features]
        targets = early_cycles_data[target_name].values
        batteries = early_cycles_data["battery"].tolist()

    return processed_features, targets, batteries, features


def evaluate_model(
    model,
    data: pd.DataFrame,
    targets: np.ndarray,
    batteries: list,
    features: list,
    target_name: str,
    output_path: Path,
    dataset_name: str,
) -> Dict[str, float]:
    """Evaluate a pre-trained model on given data."""
    predictions = model.predict(data[features])
    metrics = get_metrics(predictions, targets, batteries)

    predictions_df = pd.DataFrame({
        "battery": batteries,
        target_name: targets,
        f"predicted_{target_name}": predictions,
        "dataset": dataset_name,
    })
    output_file = output_path / f"{dataset_name}_predictions.csv"
    predictions_df.to_csv(output_file)

    return metrics


def extract_cv_results(pipeline):
    """Extract cross-validation results from a fitted pipeline."""
    cv_metrics = {}
    grid_search = pipeline.steps[-1][1]

    cv_results = grid_search.cv_results_
    cv_metrics["mean_cv_mae"] = -np.mean(cv_results["mean_test_mae"])
    cv_metrics["std_cv_mae"] = np.std(cv_results["mean_test_mae"])
    cv_metrics["mean_cv_r2"] = np.mean(cv_results["mean_test_r2"])
    cv_metrics["std_cv_r2"] = np.std(cv_results["mean_test_r2"])
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_model, best_params, cv_metrics


def train_and_evaluate(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    train_batteries: list,
    test_batteries: list,
    features: list,
    param_grid: dict,
    target_name: str,
    output_path: Path,
    cv_type: str = "kfold",
    random_state: int = 42,
    n_splits: int = 5,
    cv_repeats: int = 1,
) -> Dict[str, Any]:
    """Train Random Forest with cross-validation and evaluate."""

    cv_class = RepeatedKFold if cv_type == "repeated_kfold" else KFold
    cv = cv_class(n_splits=n_splits, shuffle=True, random_state=random_state)

    model = RandomForestRegressor()

    pipeline = generate_cross_validation_pipeline(
        regressor=model,
        parameters_grid=param_grid,
        cv=cv,
        feature_selector=None,
        feature_transformer=None,
        random_state=random_state,
        number_of_jobs=-1,
        scoring={"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error", "r2": "r2"},
        refit="mae",
    )

    # Fit the pipeline
    pipeline.fit(train_data[features], train_targets)

    # Extract results from the pipeline
    best_model, best_params, cv_metrics = extract_cv_results(pipeline)

    # Evaluate on all datasets
    train_metrics = evaluate_model(
        best_model,
        train_data,
        train_targets,
        train_batteries,
        features,
        target_name,
        output_path,
        "train",
    )

    test_metrics = evaluate_model(
        best_model,
        test_data,
        test_targets,
        test_batteries,
        features,
        target_name,
        output_path,
        "test",
    )

    # Compile all metrics
    metrics = {
        "train": train_metrics,
        "test": test_metrics,
        "best_params": best_params,
        "cv_metrics": cv_metrics,
    }

    logger.info("\nResults Summary:")
    logger.info(f"Train RMSE: {metrics['train']['rmse']:.4f}")
    logger.info(f"Train MAPE: {metrics['train']['mape']:.4f}")
    logger.info(f"Test RMSE: {metrics['test']['rmse']:.4f}")
    logger.info(f"Test MAPE: {metrics['test']['mape']:.4f}")
    logger.info(f"Test R²: {metrics['test']['r2']:.4f}")
    logger.info(f"Results saved to {output_path}")

    joblib.dump(best_model, output_path / "best_model.pkl")

    # Save feature importances
    if hasattr(best_model, "feature_importances_"):
        feature_importance_df = pd.DataFrame({
            "feature": features,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        feature_importance_df.to_csv(output_path / "feature_importances.csv")

    # Save results summary
    with open(output_path / "model_results.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


@click.command()
@click.option(
    "--train_data_path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to training data CSV",
    required=True,
)
@click.option(
    "--val_data_path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to validation data CSV",
    required=True,
)
@click.option(
    "--test_data_path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to test data CSV",
    required=True,
)
@click.option(
    "--model_path",
    type=click.Path(path_type=Path),
    help="Path to pre-trained model for evaluation mode",
)
@click.option(
    "--output_path", type=click.Path(path_type=Path), help="Path to save results", required=True
)
@click.option("--target_name", type=str, help="Name of target column", required=True)
@click.option(
    "--cycle_cutoff", type=int, help="Number of cycles to use for training", required=True
)
@click.option("--avg_cycles", type=bool, help="Whether to average cycles", default=False)
@click.option(
    "--model_parameters_grid",
    required=False,
    type=str,
    help="Model parameters grid for grid search as JSON-formatted string",
)
@click.option(
    "--cycle_window",
    type=int,
    help="Number of cycles per window (if avg_cycles=True)",
    default=None,
)
@click.option("--random_state", type=int, help="Random seed", default=42)
@click.option("--n_splits", type=int, help="Number of CV splits", default=5)
@click.option(
    "--cv_type", type=click.Choice(["kfold", "repeated_kfold"]), help="CV strategy", default="kfold"
)
@click.option("--cv_repeats", type=int, help="Number of CV repeats (for repeated_kfold)", default=1)
def main(
    train_data_path: Path,
    test_data_path: Path,
    val_data_path: Path,
    model_path: Optional[Path],
    output_path: Path,
    target_name: str,
    cycle_cutoff: int,
    model_parameters_grid: Optional[str],
    avg_cycles: bool = False,
    cycle_window: Optional[int] = None,
    random_state: int = 42,
    n_splits: int = 5,
    cv_type: str = "kfold",
    cv_repeats: int = 1,
) -> None:
    """Train and evaluate Random Forest model for battery cycle life prediction."""

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Evaluation mode (using pre-trained model)
    if model_path is not None:
        logger.info(f"Running in evaluation mode using model from {model_path}")
        model = joblib.load(model_path)

        # Load and evaluate on all available datasets
        data_paths = []
        if train_data_path:
            data_paths.append(("train", train_data_path))
        if test_data_path:
            data_paths.append(("test", test_data_path))

        all_metrics = {}
        for split_name, data_path in data_paths:
            logger.info(f"Evaluating on {split_name} data...")
            data = pd.read_csv(data_path, index_col=0)

            processed_data, targets, batteries, features = process_dataset(
                data, cycle_cutoff, target_name, avg_cycles=avg_cycles, cycle_window=cycle_window
            )

            metrics = evaluate_model(
                model=model,
                data=processed_data,
                targets=targets,
                batteries=batteries,
                features=features,
                target_name=target_name,
                output_path=output_path,
                dataset_name=split_name,
            )

            all_metrics[split_name] = metrics
            logger.info(
                f"{split_name.capitalize()} MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}"
            )

        # Save evaluation results
        with open(output_path / "evaluation_results.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        return

    # Training mode
    logger.info("Loading data for training...")

    # Load datasets
    train_data = pd.read_csv(train_data_path, index_col=0)
    val_data = pd.read_csv(val_data_path, index_col=0)
    combined_train_data = pd.concat([train_data, val_data]).drop_duplicates()

    test_data = pd.read_csv(test_data_path, index_col=0)

    logger.info(f"Combined training data shape: {combined_train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")

    logger.info("Processing datasets...")

    # Process all datasets
    train_processed, train_targets, train_batteries, features = process_dataset(
        train_data, cycle_cutoff, target_name, avg_cycles=avg_cycles, cycle_window=cycle_window
    )

    test_processed, test_targets, test_batteries, _ = process_dataset(
        test_data, cycle_cutoff, target_name, avg_cycles=avg_cycles, cycle_window=cycle_window
    )

    logger.info(f"Train samples: {len(train_batteries)}, Test samples: {len(test_batteries)}")

    if model_parameters_grid:
        param_grid = ast.literal_eval(model_parameters_grid)
    else:
        param_grid = DEFAULT_RF_PARAMS
    param_grid["random_state"] = [random_state]

    train_and_evaluate(
        train_data=train_processed,
        test_data=test_processed,
        train_targets=train_targets,
        test_targets=test_targets,
        train_batteries=train_batteries,
        test_batteries=test_batteries,
        features=features,
        param_grid=param_grid,
        target_name=target_name,
        output_path=output_path,
        random_state=random_state,
        n_splits=n_splits,
        cv_type=cv_type,
        cv_repeats=cv_repeats,
    )
