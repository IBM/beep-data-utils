"""Data handler for Beep data."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from ..configuration import MODEL_SETTINGS
from .core import Device, Experiment, ExperimentSet

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logging.basicConfig()


class BeepDataHandlerCycle:
    """DigitalTwin data handler for Beep."""

    def __init__(self):
        self.batch_chemistry_mapping = {
            "b1c": "li_ion",
            "b2c": "li_ion",
            "b3c": "li_ion",
            "b1cl": "li_ion",
            "b2cl": "li_ion",
            "b3cl": "li_ion",
            "b4cl": "li_ion",
            "b5cl": "li_ion",
            "nca": "Lithium_nickel_cobalt",
            "ncm": "Lithium_nickel_mangan",
            "na-ion": "Sodium-ion",
        }

    def get_battery_chemistry(self, cell_key: str) -> str:
        """Determine battery chemistry from cell key by matching prefixes.

        Args:
            cell_key: Unique identifier for the cell

        Returns:
            String identifying the battery chemistry
        """
        # Look through all defined prefixes and find a match
        for prefix, chemistry in self.batch_chemistry_mapping.items():
            if prefix != "default" and cell_key.lower().startswith(prefix.lower()):
                return chemistry
        return self.batch_chemistry_mapping.get("default", "unknown")

    def read_experiments(
        self,
        testing_data_path: Path,
        target_variables: List[str],
        drop_cycle_zero: bool = False,
        sampling_frequency: Optional[str] = None,
    ) -> ExperimentSet:
        """Load data.

        Args:
            testing_data_path: path to the CSV file
            target_variables: list of variables to be marked as targets
            drop_cycle_zero: whether to drop cycle zero data
            sampling_frequency: Optional sampling frequency. If "cycle", uses cycle-level aggregation
        """
        df = pd.read_csv(testing_data_path, low_memory=False)
        if drop_cycle_zero:
            df = df[df["cycle_number"] != 0].copy()
        if sampling_frequency == "cycle":
            logger.info("Performing cycle based averaging ")
            preserve_features = [
                "charge_policy_Q1",
                "charge_policy_Q2",
                "charge_policy_Q3",
                "charge_policy_Q4",
            ]
            preserve_features.extend(target_variables)
            df = preprocess_battery_data(df, preserve_features=preserve_features)

        for var in target_variables:
            if var in df.columns and f"target_{var}" not in df.columns:
                df[f"target_{var}"] = df[var].astype(float)
            else:
                df[f"target_{var}"] = MODEL_SETTINGS.unknown_numerical_value

        cells = df["cell_key"].unique()
        experiments = []

        all_columns = df.columns.tolist()
        base_columns = [
            col
            for col in all_columns
            if not col.startswith("target_") and col not in ["cell_key"] + target_variables
        ]

        # Add new target columns to operating conditions list
        target_columns = [f"target_{var}" for var in target_variables]
        operating_columns = list(set(base_columns + target_columns))

        for cell_key in tqdm(cells, desc="Processing cells"):
            cell_data = df[df["cell_key"] == cell_key].copy()
            chemistry = self.get_battery_chemistry(cell_key)

            charge_policy_columns = [
                col for col in cell_data.columns if col.startswith("charge_policy_Q")
            ]
            if charge_policy_columns:
                charge_policy = {
                    "Q1": cell_data["charge_policy_Q1"].iloc[0],
                    "Q2": cell_data["charge_policy_Q2"].iloc[0],
                    "Q3": cell_data["charge_policy_Q3"].iloc[0],
                    "Q4": cell_data["charge_policy_Q4"].iloc[0],
                }
            else:
                charge_policy = cell_data["charge_policy"].iloc[0]

            device_data = pd.Series({
                "device": cell_key,
                "charge_policy": charge_policy,
                "discharge_policy": cell_data["discharge_policy"].iloc[0],
                "batch_number": cell_data["batch_number"].iloc[0]
                if "batch_number" in cell_data.columns
                else None,
                "battery_chemistry": chemistry,
            })

            device = Device(device_data=device_data)

            available_columns = [col for col in operating_columns if col in cell_data.columns]
            operating_conditions = cell_data[available_columns].copy()

            experiments.append(Experiment(device=device, operating_conditions=operating_conditions))

        return ExperimentSet(experiments)


def preprocess_battery_data(df, preserve_features=None):
    """Preprocess battery data by aggregating at cycle level.
    For each numeric column, calculates specified statistics at the cycle level.
    Returns:
        pd.DataFrame: Aggregated data at the cycle level
    """
    feature_stats_mapping = {
        "default": ["min", "max", "mean", "median", "std"],
        "current": ["min", "max", "mean", "median", "std"],
        "voltage": ["min", "max", "mean", "median", "std"],
        "temperature": ["min", "max", "mean", "median", "std"],
        "discharge_capacity": ["max"],
        "charge_capacity": ["max"],
    }

    if preserve_features is None:
        preserve_features = []

    id_cols = ["cell_key", "cycle_number"]
    default_preserve = ["discharge_policy", "batch_number"]
    default_preserve = [col for col in default_preserve if col in df.columns]

    charge_policy_cols = [col for col in df.columns if col.startswith("charge_policy")]

    all_preserve_cols = id_cols + default_preserve + charge_policy_cols + preserve_features
    all_preserve_cols = list(set(all_preserve_cols))
    agg_dict = {}

    for col in all_preserve_cols:
        if col in df.columns and col not in id_cols:
            agg_dict[col] = "first"

    for col in df.columns:
        if col not in all_preserve_cols and col != "time":  # Skip time
            if col in feature_stats_mapping:
                stats = feature_stats_mapping[col]
            else:
                stats = feature_stats_mapping.get("default")

            agg_dict[col] = stats

    # Perform the aggregation efficiently using pandas
    result = df.groupby(["cell_key", "cycle_number"]).agg(agg_dict)

    result.columns = [f"{col}_{agg}" if agg != "first" else col for col, agg in result.columns]
    result = result.reset_index()
    result["time"] = result[
        "cycle_number"
    ]  # Take the cycle life as the time, as we average over a cycle

    return result
