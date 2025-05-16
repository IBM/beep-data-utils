"""Files from here: https://data.matr.io/1/projects/5d80e633f405260001c0b60a"""

import pickle
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd


def parse_charge_policy(policy: str, num_quantiles: int = 4) -> List[float]:
    """Parse charge policy string into list of C-rates for 4 quartiles.

    Handles multiple formats. Similar as in beep_Dataset_creation_cli.py:
    - Simple C-rate: "4.8C" -> [4.8, 4.8, 4.8, 4.8]
    - Percentage-based: "3.6C(80%)-1.6C" -> [3.6, 3.6, 3.6, 1.6]
    - Direct quartiles: "4-6-5.6-4.2" -> [4.0, 6.0, 5.6, 4.2]
    """
    # I know its strange bt this newstructure originates from the pickle files.
    policy = str(policy).replace("charge_policy_", "").replace("-newstructure", "").replace(" ", "")

    # Handle direct quartile format (e.g., "4-6-5.6-4.2")
    if "-" in policy and "C" not in policy:
        parts = policy.split("-")
        if len(parts) == num_quantiles:
            return [float(rate) for rate in parts]

    if "(" not in policy and "C" in policy:
        rate = float(policy.replace("C", ""))
        return [rate] * num_quantiles

    all_rates = np.zeros(100)
    current_position = 0
    stages = policy.split("-")

    for stage in stages[:-1]:
        if "(" not in stage:
            continue

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


def expand_charge_policy_columns(row: dict) -> dict:
    """
    Charge policy into separate columns for each quartile.

    Args:
        row: Dictionary containing row data with 'charge_policy' key

    Returns:
        Modified row with Q1-Q4 columns replacing charge_policy
    """
    rates = parse_charge_policy(row["charge_policy"])
    new_row = row.copy()
    del new_row["charge_policy"]

    for i, rate in enumerate(rates, 1):
        new_row[f"charge_policy_Q{i}"] = rate

    return new_row


def write_chunk_with_policy_columns(data: list, output_path: Path, is_first_chunk: bool) -> None:
    """Write a chunk of data to the output CSV file with separate charge policy columns."""
    expanded_data = [expand_charge_policy_columns(row) for row in data]
    df = pd.DataFrame(expanded_data)
    df.to_csv(output_path, mode="w" if is_first_chunk else "a", header=is_first_chunk, index=False)


def extract_targets(batches: list, output_dir: Path, target_cols: List[str]) -> None:
    """Extract target variables from preprocessed batches and save to CSV."""
    target_list = []

    for batch_num, batch in enumerate(batches, 1):
        if batch_num != 5:
            continue

        for cell_key, cell_data in batch.items():
            cell_targets = {"cell_key": cell_key}
            for target_col in target_cols:
                value = cell_data[target_col]
                if isinstance(value, np.ndarray):
                    value = value[0][0] if value.size > 0 else None
                cell_targets[target_col] = value
            target_list.append(cell_targets)

    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.DataFrame(target_list)
    if not test_df.empty:
        test_df = test_df.sort_values("cell_key")
        test_path = output_dir / "targets.csv"
        if test_path.exists():
            existing_df = pd.read_csv(test_path)
            combined_df = pd.concat([existing_df, test_df], ignore_index=True)
            combined_df = combined_df.sort_values("cell_key").drop_duplicates()
            combined_df.to_csv(test_path, index=False)
        else:
            test_df.to_csv(test_path, index=False)

        click.echo("\nTesting Target Data Summary:")
        click.echo("-" * 50)
        click.echo(f"Total cells: {len(test_df)}")
        for col in target_cols:
            if pd.api.types.is_numeric_dtype(test_df[col]):
                click.echo(f"\n{col} statistics (Testing):")
                click.echo(f"Minimum: {test_df[col].min()}")
                click.echo(f"Maximum: {test_df[col].max()}")
                click.echo(f"Average: {test_df[col].mean():.2f}")

    click.echo(f"\nSaved target data to: {output_dir}")


def extract_train_test_data(batches: list, output_dir: Path, chunk_size: int = 1000000) -> None:
    """Extract and save training and testing datasets separately."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_secondary_beep_dataset_quartile_charge_policy.csv"
    test_path = output_dir / "test_secondary_beep_dataset_quartile_charge_policy.csv"

    train_data = []
    test_data = []
    total_processed_train = 0
    total_processed_test = 0

    # First process batches 1-4 (training data)
    for batch_num in range(1, 5):
        click.echo(f"Processing training batch {batch_num}...")
        batch = batches[batch_num - 1]

        for cell_key, cell_data in batch.items():
            cycles = cell_data["cycles"]

            for cycle_num, cycle_data in cycles.items():
                cycle_length = len(cycle_data["I"])
                for idx in range(cycle_length):
                    row = {
                        "cell_key": cell_key,
                        "cycle_number": int(cycle_num),
                        "time": float(cycle_data["t"][idx]),
                        "discharge_policy": 400,
                        "charge_policy": cell_data["charge_policy"],
                        "current": float(cycle_data["I"][idx]),
                        "voltage": float(cycle_data["V"][idx]),
                        "charge_capacity": float(cycle_data["Qc"][idx]),
                        "discharge_capacity": float(cycle_data["Qd"][idx]),
                        "temperature": float(cycle_data["T"][idx]),
                        "batch_number": batch_num,
                    }

                    if "dQdV" in cycle_data and idx < len(cycle_data["dQdV"]):
                        row["discharge_dQdV"] = float(cycle_data["dQdV"][idx])

                    train_data.append(row)
                    total_processed_train += 1

                    if len(train_data) >= chunk_size:
                        write_chunk_with_policy_columns(
                            train_data, train_path, total_processed_train == len(train_data)
                        )
                        train_data = []

            if total_processed_train % 100000 == 0:
                click.echo(f"Processed {total_processed_train} training measurements...")

    # Then process batch 5 (testing data)
    click.echo("Processing testing batch (batch 5)...")
    batch = batches[4]

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
                    "batch_number": 5,
                }

                if "dQdV" in cycle_data and idx < len(cycle_data["dQdV"]):
                    row["discharge_dQdV"] = float(cycle_data["dQdV"][idx])

                test_data.append(row)
                total_processed_test += 1

                if len(test_data) >= chunk_size:
                    write_chunk_with_policy_columns(
                        test_data, test_path, total_processed_test == len(test_data)
                    )
                    test_data = []

        if total_processed_test % 100000 == 0:
            click.echo(f"Processed {total_processed_test} testing measurements...")

    if train_data:
        write_chunk_with_policy_columns(
            train_data, train_path, total_processed_train == len(train_data)
        )
    if test_data:
        write_chunk_with_policy_columns(
            test_data, test_path, total_processed_test == len(test_data)
        )

    click.echo("\nTraining Dataset Summary:")
    display_summary(train_path)

    click.echo("\nTesting Dataset Summary:")
    display_summary(test_path)


def load_batch_data(data_path: Path) -> list:
    """Load and preprocess batch data from pickle files."""
    batch1 = pickle.load(open(data_path / "batch1_2_ds.pkl", "rb"))
    batch2 = pickle.load(open(data_path / "batch2_2_ds.pkl", "rb"))
    batch3 = pickle.load(open(data_path / "batch3_2_ds.pkl", "rb"))
    batch4 = pickle.load(open(data_path / "batch4_2_ds.pkl", "rb"))
    batch5 = pickle.load(open(data_path / "batch5_2_ds.pkl", "rb"))

    return [batch1, batch2, batch3, batch4, batch5]


def display_summary(dataset_path: Path) -> None:
    """Display summary statistics for the generated dataset."""
    df = pd.read_csv(dataset_path)
    click.echo("-" * 50)
    click.echo(f"Total measurements: {len(df)}")
    click.echo(f"Unique cells: {df['cell_key'].nunique()}")
    click.echo(f"Unique cycles: {df['cycle_number'].nunique()}")
    click.echo(f"Total size in memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


def write_domain_files(batches: list) -> None:
    """Append cell indices to existing domain classification files."""
    in_domain_path = "in_domain.txt"
    out_domain_path = "out_of_domain.txt"

    in_domain_cells = []
    out_domain_cells = []

    # Process batches 1-4 for in_domain
    for batch_num in range(4):
        batch = batches[batch_num]
        for cell_key in batch.keys():
            in_domain_cells.append(f"{cell_key}")

    # Process batch 5 for out_of_domain
    batch5 = batches[4]
    for cell_key in batch5.keys():
        out_domain_cells.append(f"{cell_key}")

    # Sort the cell indices for consistency
    in_domain_cells.sort()
    out_domain_cells.sort()

    # Append to existing files with a newline before new content
    with open(in_domain_path, "a") as f:
        f.write("\n" + "\n".join(in_domain_cells))

    with open(out_domain_path, "a") as f:
        f.write("\n" + "\n".join(out_domain_cells))

    click.echo("\nAppended to domain classification files:")
    click.echo(f"In-domain cells: {len(in_domain_cells)} cells added")
    click.echo(f"Out-of-domain cells: {len(out_domain_cells)} cells added")


@click.command()
@click.option(
    "--pickle-data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory containing batch1_2_ds.pkl through batch5_2_ds.pkl",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Directory where train and test datasets will be saved",
)
@click.option(
    "--target-columns",
    type=str,
    required=False,
    help="Comma-separated list of target column names to extract",
)
@click.option(
    "--chunk-size",
    type=int,
    default=1000000,
    show_default=True,
    help="Number of rows to process at once for memory efficiency",
)
@click.option(
    "--add_cells_to_domain_txt_file",
    type=bool,
    default=False,
    show_default=True,
    help="If the cycle index should be added to the in/out domain file",
)
def generate_sequence_dataset(
    pickle_data_path: Path,
    output_dir: Path,
    chunk_size: int,
    add_cells_to_domain_txt_file: bool,
    target_columns: Optional[str],
) -> None:
    """Generate train and test datasets from battery data."""
    output_dir = Path(output_dir)

    click.echo("Loading and preprocessing batch data...")
    batches = load_batch_data(Path(pickle_data_path))

    if target_columns:
        target_cols = [col.strip() for col in target_columns.split(",")]
        click.echo(f"Extracting target variables: {', '.join(target_cols)}")
        extract_targets(batches, output_dir, target_cols)

    if add_cells_to_domain_txt_file:
        click.echo("Appending to domain classification files...")
        write_domain_files(batches)

    click.echo("Generating train and test datasets...")
    extract_train_test_data(batches, output_dir, chunk_size)

    click.echo("\nDataset generation complete!")
    click.echo(f"Output directory: {output_dir}")
