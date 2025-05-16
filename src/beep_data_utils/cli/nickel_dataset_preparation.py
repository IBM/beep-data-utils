"""Na-ion battery dataset preprocessing CLI. https://zenodo.org/records/7981011"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import click
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ProtocolMetadata:
    """Metadata for each protocol including charge and discharge policies."""

    name: str
    charge_policy_Q1: float
    charge_policy_Q2: float
    charge_policy_Q3: float
    charge_policy_Q4: float
    discharge_policy: str
    description: str


def get_protocol_definitions() -> Dict[str, ProtocolMetadata]:
    """Define protocols with quartile-based charge policies."""
    return {
        "C20Form": ProtocolMetadata(
            name="Standard C/20 formation",
            charge_policy_Q1=0.05,
            charge_policy_Q2=0.05,
            charge_policy_Q3=0.05,
            charge_policy_Q4=0.05,
            discharge_policy="005100",
            description="Standard C/20 formation protocol",
        ),
        "C20Form 3xFormcycle": ProtocolMetadata(
            name="Triple formation cycle",
            charge_policy_Q1=0.05,
            charge_policy_Q2=0.05,
            charge_policy_Q3=0.05,
            charge_policy_Q4=0.05,
            discharge_policy="005100",
            description="Triple C/20 formation cycle",
        ),
        "Pulsed Formation": ProtocolMetadata(
            name="Pulsed charging formation",
            charge_policy_Q1=0.17,
            charge_policy_Q2=0.17,
            charge_policy_Q3=0.17,
            charge_policy_Q4=0.17,
            discharge_policy="005100",
            description="Pulsed charging formation protocol",
        ),
        "C20Formation C2 Cycle": ProtocolMetadata(
            name="C/20 formation with C/2 cycling",
            charge_policy_Q1=0.05,
            charge_policy_Q2=0.05,
            charge_policy_Q3=0.50,
            charge_policy_Q4=0.50,
            discharge_policy="005100",
            description="C/20 formation followed by C/2 cycling",
        ),
    }


def load_and_harmonize_file(file_path: Path, protocol: ProtocolMetadata) -> pd.DataFrame:
    """Load and harmonize a single test file to match required format."""
    usecols = [
        "Test_Time(s)",
        "Cycle_Index",
        "Current(A)",
        "Voltage(V)",
        "Charge_Capacity(Ah)",
        "Discharge_Capacity(Ah)",
        "dQ/dV(Ah/V)",
    ]

    df = pd.read_csv(file_path, usecols=usecols)

    parts = file_path.stem.split("_")
    test_num = parts[1]
    cell_num = parts[2].replace("cell", "")

    harmonized = pd.DataFrame({
        "cell_key": f"na-ion_{test_num}_cell{cell_num}",
        "cycle_number": df["Cycle_Index"],
        "time": df["Test_Time(s)"],
        "charge_policy_Q1": protocol.charge_policy_Q1,
        "charge_policy_Q2": protocol.charge_policy_Q2,
        "charge_policy_Q3": protocol.charge_policy_Q3,
        "charge_policy_Q4": protocol.charge_policy_Q4,
        "discharge_policy": 100,
        "current": df["Current(A)"],
        "voltage": df["Voltage(V)"],
        "temperature": pd.Series(np.nan, index=df.index, dtype="float64"),
        "charge_capacity": df["Charge_Capacity(Ah)"],
        "discharge_capacity": df["Discharge_Capacity(Ah)"],
        "discharge_dQdV": df["dQ/dV(Ah/V)"],
    })

    return harmonized


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing the Na-ion battery protocol folders",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Path where the processed dataset will be saved",
)
def main(data_dir: Union[str, Path], output_path: Union[str, Path]):
    """Process Na-ion battery cycling data and generate standardized dataset, to fit into this BEEP FM."""
    # Convert string paths to Path objects
    data_dir = Path(str(data_dir))
    output_path = Path(str(output_path))

    protocols = get_protocol_definitions()
    all_data = []
    total_files = 0

    logger.info("Processing Na-ion battery protocols:")

    for folder_name, protocol in protocols.items():
        protocol_path = data_dir / folder_name
        if not protocol_path.exists():
            continue

        csv_files = list(protocol_path.glob("**/*Channel_*_Wb_*.CSV"))
        logger.info(f"\n{protocol.name}:")
        logger.info(f"Found {len(csv_files)} test files")

        for file_path in csv_files:
            try:
                df = load_and_harmonize_file(file_path, protocol)
                all_data.append(df)
                total_files += 1
                logger.info(f"Processed {file_path.name}")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue

    if not all_data:
        logger.error("No data was processed successfully.")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.sort_values(["cell_key", "time"], inplace=True)

    # Ensure consistent column order
    columns = [
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

    # Save outputs
    combined_df[columns].to_csv(output_path, index=False, float_format="%.12g")

    logger.info("\nDataset Summary:")
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Total rows: {len(combined_df)}")
    logger.info(f"Unique cells: {combined_df['cell_key'].nunique()}")
    logger.info(f"Output saved to: {output_path}")
