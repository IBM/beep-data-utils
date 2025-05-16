import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import pandas as pd


def parse_charge_policy(policy: str, num_quantiles: int = 4) -> List[float]:
    """Parse charge policy string into list of C-rates for 4 quartiles.

    Examples:
    - "3.6C(80%)-1.6C" -> First rate for 80%, second rate for 20% [3.6, 3.6, 3.6, 2.0]
    - "5.4C(40%)-3.6C" -> First rate for 40%, second rate for 60% [5.4, 4.68, 3.6, 3.6]
    - "5.4C(15%)-1.6C(75%)-2C" -> [3.88, 1.6, 1.6, 2.0]
    """
    policy = str(policy).replace("charge_policy_", "").replace("-newstructure", "").replace(" ", "")

    all_rates = np.zeros(100)
    current_position = 0
    stages = policy.split("-")

    for stage in stages[:-1]:
        rate_part, percentage_part = stage.split("C(")
        rate = float(rate_part)
        percentage = int(float(percentage_part.replace("%)", "")))

        all_rates[current_position:percentage] = rate
        current_position = percentage

    final_rate = float(stages[-1].replace("C", ""))
    all_rates[current_position:] = final_rate

    window_size = 100 // num_quantiles
    quantile_rates = []

    for i in range(num_quantiles):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        quantile_rate = np.mean(all_rates[start_idx:end_idx])
        quantile_rates.append(round(quantile_rate, 3))

    return quantile_rates


def expand_charge_policy_columns(row: Dict[Any, Any], num_quantiles: int = 4) -> Dict[str, float]:
    """
    Expand charge policy into separate columns for each quartile.
    """
    rates = parse_charge_policy(row["charge_policy"], num_quantiles)
    new_row = row.copy()
    del new_row["charge_policy"]

    for i, rate in enumerate(rates, 1):
        new_row[f"charge_policy_Q{i}"] = rate

    return new_row


# as in https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb
def preprocess_batch_data(batch1: Dict, batch2: Dict, batch3: Dict) -> Tuple[Dict, Dict, Dict]:
    """Preprocess and clean batch data according to specified rules."""
    # Remove batteries from batch1 that do not reach 80% capacity
    batch1_removal = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
    batch3_removals = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]

    for key in batch1_removal:
        del batch1[key]

    # Batch 1 and 2 preprocessing
    batch2_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
    batch1_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
    add_len = [662, 981, 1060, 208, 482]

    # Merge specific cells from batch2 into batch1
    for i, bk in enumerate(batch1_keys):
        batch1[bk]["cycle_life"] = batch1[bk]["cycle_life"] + add_len[i]

        # Update summary data
        for j in batch1[bk]["summary"].keys():
            if j == "cycle":
                batch1[bk]["summary"][j] = np.hstack((
                    batch1[bk]["summary"][j],
                    batch2[batch2_keys[i]]["summary"][j] + len(batch1[bk]["summary"][j]),
                ))
            else:
                batch1[bk]["summary"][j] = np.hstack((
                    batch1[bk]["summary"][j],
                    batch2[batch2_keys[i]]["summary"][j],
                ))

        last_cycle = len(batch1[bk]["cycles"].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]["cycles"].keys()):
            batch1[bk]["cycles"][str(last_cycle + j)] = batch2[batch2_keys[i]]["cycles"][jk]

    for key in batch2_keys:
        del batch2[key]

    for channel in batch3_removals:
        if channel in batch3:
            del batch3[channel]

    return batch1, batch2, batch3


def load_batch_data(data_path: Path) -> list:
    """Load and preprocess batch data from pickle files."""
    batch1 = pickle.load(open(data_path / "batch1.pkl", "rb"))
    batch2 = pickle.load(open(data_path / "batch2.pkl", "rb"))
    batch3 = pickle.load(open(data_path / "batch3.pkl", "rb"))

    batch1, batch2, batch3 = preprocess_batch_data(batch1, batch2, batch3)
    return [batch1, batch2, batch3]


@click.command()
@click.option(
    "--pickle-data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory containing batch1.pkl, batch2.pkl, and batch3.pkl",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Path where the uniformed dataset CSV will be saved",
)
@click.option(
    "--target-columns",
    type=str,
    default="cycle_life",
    help="Comma-separated list of target column names to extract",
)
@click.option(
    "--chunk-size",
    type=int,
    default=1000000,
    show_default=True,
    help="Number of rows to process at once for memory efficiency",
)
def generate_sequence_dataset(
    pickle_data_path: Path,
    output_path: Path,
    target_columns: str,
    chunk_size: int,
) -> None:
    """
    Generate a uniformed sequence dataset from battery data.

    This CLI script processes battery test data from
    https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 manipulated through these scrips:
    https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/BuildPkl_Batch1.ipynb
    and creates a uniformed dataset containing detailed cycle-by-cycle measurements.
    """
    click.echo("Loading and preprocessing batch data...")
    batches = load_batch_data(Path(pickle_data_path))

    # Parse target columns
    target_cols = [col.strip() for col in target_columns.split(",")]
    click.echo(f"Extracting target variables: {', '.join(target_cols)}")

    all_data = []
    total_processed = 0

    for batch_num, batch in enumerate(batches, 1):
        click.echo(f"Processing batch {batch_num}...")

        for cell_key, cell_data in batch.items():
            cycles = cell_data["cycles"]
            cycle_life = int(cell_data["cycle_life"][0][0])

            for cycle_num, cycle_data in cycles.items():
                cycle_length = len(cycle_data["I"])

                for idx in range(cycle_length):
                    row = {
                        "cell_key": cell_key,
                        "cycle_number": int(cycle_num),
                        "time": float(cycle_data["t"][idx]),
                        "cycle_life": cycle_life,
                        "discharge_policy": 400,
                        "charge_policy": cell_data["charge_policy"],
                        "current": float(cycle_data["I"][idx]),
                        "voltage": float(cycle_data["V"][idx]),
                        "charge_capacity": float(cycle_data["Qc"][idx]),
                        "discharge_capacity": float(cycle_data["Qd"][idx]),
                        "temperature": float(cycle_data["T"][idx]),
                        "batch_number": batch_num,
                        "discharge_dQdV": None,  # Default value
                    }

                    if "dQdV" in cycle_data and idx < len(cycle_data["dQdV"]):
                        row["discharge_dQdV"] = float(cycle_data["dQdV"][idx])

                    all_data.append(row)
                    total_processed += 1

                    # Write chunk when size threshold is reached
                    if len(all_data) >= chunk_size:
                        write_chunk(all_data, output_path, total_processed == len(all_data))
                        all_data = []

            if total_processed % 100000 == 0:
                click.echo(f"Processed {total_processed} measurements...")

    # Write any remaining data
    if all_data:
        write_chunk(all_data, output_path, total_processed == len(all_data))

    display_summary(output_path)


def write_chunk(data: list, output_path: Path, is_first_chunk: bool) -> None:
    """Write a chunk of data to the output CSV file with separate charge policy columns."""
    # Expand charge policy into columns for each row
    expanded_data = [expand_charge_policy_columns(row) for row in data]

    df = pd.DataFrame(expanded_data)
    df.to_csv(output_path, mode="w" if is_first_chunk else "a", header=is_first_chunk, index=False)


def display_summary(dataset_path: Path) -> None:
    """Display summary statistics for the generated dataset."""
    df = pd.read_csv(dataset_path)
    click.echo("\nDataset Summary:")
    click.echo("-" * 50)
    click.echo(f"Total measurements: {len(df)}")
    click.echo(f"Unique cells: {df['cell_key'].nunique()}")
    click.echo(f"Unique cycles: {df['cycle_number'].nunique()}")
    click.echo(f"Total size in memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
