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


class BeepDataHandler:
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
        return self.batch_chemistry_mapping["default"]

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
            sampling_frequency: Optional sampling frequency for this specific dataset
        """
        df = pd.read_csv(testing_data_path, low_memory=False)
        if drop_cycle_zero:
            df = df[df["cycle_number"] != 0].copy()

        if sampling_frequency is not None:
            df["_timestep"] = pd.to_datetime(df["time"], unit="s")
            df = (
                df.groupby("cell_key")
                .apply(lambda x: x.resample(sampling_frequency, on="_timestep").first())
                .reset_index(drop=True)
            )
            df.dropna(subset=["time"], inplace=True)

        for var in target_variables:
            if var in df.columns and f"target_{var}" not in df.columns:
                df[f"target_{var}"] = df[var].astype(float)
            else:
                df[f"target_{var}"] = MODEL_SETTINGS.unknown_numerical_value

        cells = df["cell_key"].unique()
        experiments = []

        base_columns = [
            "cycle_number",
            "current",
            "voltage",
            "discharge_policy",
            "charge_policy",
            "charge_capacity",
            "discharge_capacity",
            "temperature",
            "time",
            "discharge_dQdV",
        ]

        base_columns = [col for col in base_columns if col not in target_variables]

        for var in target_variables:
            if var in df.columns:
                df[f"target_{var}"] = df[var].astype(float)
                if var in df.columns:
                    df = df.drop(columns=[var])
            else:
                df[f"target_{var}"] = MODEL_SETTINGS.unknown_numerical_value

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
                "initial_capacity": cell_data[cell_data["cycle_number"] == 1][
                    "discharge_capacity"
                ].max(),
                "battery_chemistry": chemistry,
            })

            device = Device(device_data=device_data)

            # Base columns plus charge policy quartiles and target columns
            base_columns = [
                "cycle_number",
                "current",
                "voltage",
                "charge_capacity",
                "discharge_capacity",
                "temperature",
                "time",
                "discharge_dQdV",
            ]

            target_columns = [f"target_{var}" for var in target_variables]
            operating_columns = base_columns + charge_policy_columns + target_columns

            operating_conditions = cell_data[
                [col for col in operating_columns if col in cell_data.columns]
            ].copy()

            experiments.append(Experiment(device=device, operating_conditions=operating_conditions))

        return ExperimentSet(experiments)
