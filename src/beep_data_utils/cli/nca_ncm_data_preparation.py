"""Preparation for https://zenodo.org/records/6405084"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import click
import pandas as pd
from loguru import logger

from ..configuration import MODEL_SETTINGS


class DatasetConfig(TypedDict):
    path: Path
    type: str


def convert_charge_rate(rate_str):
    """Convert charge rate string to numeric value.

    Args:
        rate_str: String representation of charge rate (e.g., "025", "05", "1")

    Returns:
        float: Charge rate as a floating point value
    """
    # I know its ugly, but does the job.
    charge_rates = {"025": 0.25, "05": 0.5, "1": 1.0, "2": 2.0, "4": 4.0}

    # Use the mapping if available, otherwise convert based on string format
    if rate_str in charge_rates:
        return charge_rates[rate_str]
    else:
        return float(rate_str) / 100.0 if len(rate_str) > 1 else float(rate_str)


def parse_charge_policy(
    filename: str,
) -> Tuple[Union[float, float], ...]:
    """Parse charge policy from filename to extract charge rates for each quartile.

    Args:
        filename: Battery filename containing charge policy information

    Returns:
        Tuple of charge rates for each quartile of the charging phase
    """
    # Handle all filename formats: CYX-Y_Z or CYX-Y/Z
    pattern = r"CY\d+-(\d+[\d.]*)[\/_].*\.csv"
    match = re.match(pattern, filename)

    if not match:
        unknown_value = MODEL_SETTINGS.unknown_numerical_value
        return (unknown_value, unknown_value, unknown_value, unknown_value)

    # Extract the charge rate string
    charge_rate_str = match.group(1)

    # Convert charge rate string to actual C-rate value
    charge_rate = convert_charge_rate(charge_rate_str)

    # Return the same charge rate for all quartiles (since it's constant in this dataset)
    return charge_rate, charge_rate, charge_rate, charge_rate


def expand_charge_policy_columns(row: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Expand charge policy into separate columns for each quartile.

    Args:
        row: Dictionary containing row data
        filename: Original filename containing charge policy information

    Returns:
        Updated row dictionary with quartile-based charge policy columns
    """
    rates = parse_charge_policy(filename)
    new_row = row.copy()
    new_row.pop("charge_policy", None)

    for i, rate in enumerate(rates, 1):
        new_row[f"charge_policy_Q{i}"] = rate

    return new_row


def calculate_dq_dv(group: pd.DataFrame) -> pd.Series:
    """Calculate dQ/dV for a discharge cycle.

    Args:
        group: DataFrame group containing one complete cycle
        https://www.neware.net/news/lithium-ion-battery-dq-dv-analysis/230/100.html

    Returns:
        Series of dQ/dV values aligned with the original data
    """
    df = group.copy()
    discharge_mask = df["current"] < 0
    discharge_data = df[discharge_mask].copy()

    if len(discharge_data) > 1:
        discharge_data = discharge_data.sort_values("voltage", ascending=False)
        dQ = discharge_data["discharge_capacity"].diff()
        dV = discharge_data["voltage"].diff()

        dQdV = pd.Series(index=discharge_data.index)
        mask = abs(dV) > 1e-6
        dQdV[mask] = dQ[mask] / dV[mask]

        result = pd.Series(index=group.index)
        result[discharge_mask] = dQdV
        result[~discharge_mask] = 0

        return result
    else:
        return pd.Series(0, index=group.index)


def parse_filename(filename: str) -> Optional[Dict]:
    """Parse battery filename to extract metadata.

    Args:
        filename: Filename string to parse

    Returns:
        Dictionary containing parsed metadata or None if parsing fails
    """
    # Match all valid formats of CYX-Y_Z-#N.csv or CYX-Y/Z-#N.csv
    pattern = r"CY(\d+)-(\d+[\d.]*)[/_](\d+)-#(\d+)\.csv"
    match = re.match(pattern, filename)

    if not match:
        logger.error("does not match the pattern, this should not happen")
        return None

    temperature, charge_rate_str, discharge_rate, cell_num = match.groups()

    charge_rate = convert_charge_rate(charge_rate_str)

    return {
        "temperature": int(temperature),
        "charge_rate": charge_rate,
        "discharge_rate": float(discharge_rate),
        "cell_number": int(cell_num),
    }


def process_battery_file(file_path: Path, battery_type: str) -> Optional[pd.DataFrame]:
    """Process a single battery data file.

    Args:
        file_path: Path to the battery data file
        battery_type: Type of battery (nca, ncm, ncmca)

    Returns:
        Processed DataFrame or None if processing fails
    """
    metadata = parse_filename(file_path.name)
    if not metadata:
        return None

    df = pd.read_csv(file_path)
    output_df = pd.DataFrame()

    rate_code = f"{int(metadata['charge_rate'] * 100):03d}"
    cell_key = f"{battery_type}{metadata['temperature']}{rate_code}{metadata['cell_number']:02d}"

    output_df["cell_key"] = pd.Series([cell_key] * len(df))
    output_df["time"] = df["time/s"]
    output_df["cycle_number"] = df["cycle number"]
    output_df["discharge_policy"] = metadata["discharge_rate"]

    temp_dict = {
        "cell_key": cell_key,
        "time": df["time/s"].iloc[0],
        "cycle_number": df["cycle number"].iloc[0],
        "discharge_policy": metadata["discharge_rate"],
    }
    expanded_policy = expand_charge_policy_columns(temp_dict, file_path.name)

    for key, value in expanded_policy.items():
        if key.startswith("charge_policy_Q"):
            output_df[key] = value

    output_df["current"] = df["<I>/mA"] / 1000.0  # Convert mA to A
    output_df["voltage"] = df["Ecell/V"]
    output_df["charge_capacity"] = df["Q charge/mA.h"] / 1000.0  # mAh to Ah
    output_df["discharge_capacity"] = df["Q discharge/mA.h"] / 1000.0  # mAh to Ah
    output_df["temperature"] = metadata["temperature"]

    output_df["discharge_dQdV"] = (
        output_df.groupby("cycle_number").apply(calculate_dq_dv).reset_index(level=0, drop=True)
    )

    required_columns = [
        "cell_key",
        "cycle_number",
        "time",
        "charge_policy_Q1",
        "charge_policy_Q2",
        "charge_policy_Q3",
        "charge_policy_Q4",
        "discharge_policy",
        "current",
        "voltage",
        "charge_capacity",
        "discharge_capacity",
        "temperature",
        "discharge_dQdV",
    ]

    return output_df[required_columns]


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing the battery dataset folders",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Path where the processed dataset will be saved",
)
def main(data_dir: str, output_path: str):
    """Process battery cycling data and generate standardized dataset, to fit into this BEEP FM."""
    # Convert string paths to Path objects after receiving them from click
    data_dir_path = Path(data_dir)
    output_path_path = Path(output_path)

    # Create the datasets list with proper typing
    datasets: List[DatasetConfig] = [
        {"path": data_dir_path / "Dataset_1_NCA_battery", "type": "nca"},
        {"path": data_dir_path / "Dataset_2_NCM_battery", "type": "ncm"},
        {"path": data_dir_path / "Dataset_3_NCM_NCA_battery", "type": "ncmca"},
    ]

    all_data = []

    # Process each dataset
    for dataset in datasets:
        dataset_path = dataset["path"]
        if dataset_path.exists():
            logger.info(f"Processing dataset: {dataset_path}")
            csv_files = list(dataset_path.glob("*.csv"))

            for file_path in csv_files:
                df = process_battery_file(file_path, dataset["type"])
                if df is not None:
                    all_data.append(df)
                    logger.info(f"Processed file: {file_path.name}")

    if not all_data:
        logger.error("No data was processed")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_path_path, index=False)

    logger.info(f"Processed data saved to {output_path_path}")
    logger.info(f"Total records: {len(final_df)}")
