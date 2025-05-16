from pathlib import Path
from typing import List

import click
import pandas as pd


def extract_targets_from_csv(input_csv: Path, output_path: Path, target_cols: List[str]) -> None:
    """
    Extract target variables from input CSV and append to existing targets file.

    Args:
        input_csv: Path to the input CSV containing the data
        output_path: Path where the targets CSV will be saved/updated
        target_cols: List of target column names to extract
    """
    df = pd.read_csv(input_csv)
    target_list = []
    for cell_key in df["cell_key"].unique():
        cell_data = df[df["cell_key"] == cell_key].iloc[0]
        cell_targets = {"cell_key": cell_key}

        for target_col in target_cols:
            if target_col in cell_data:
                cell_targets[target_col] = cell_data[target_col]
            else:
                click.echo(f"Warning: Target column '{target_col}' not found in the data")
                cell_targets[target_col] = None

        target_list.append(cell_targets)

    new_targets_df = pd.DataFrame(target_list)

    # Check if targets.csv already exists
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, new_targets_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["cell_key"]).sort_values("cell_key")
        combined_df.to_csv(output_path, index=False)
        click.echo(f"Updated existing targets file: {output_path}")
    else:
        # If file doesn't exist, just write the new data
        new_targets_df.sort_values("cell_key").to_csv(output_path, index=False)
        click.echo(f"Created new targets file: {output_path}")

    click.echo("\nTarget Data Summary:")
    click.echo("-" * 50)
    click.echo(f"Total cells processed: {len(new_targets_df)}")
    for col in target_cols:
        if col in new_targets_df.columns and pd.api.types.is_numeric_dtype(new_targets_df[col]):
            click.echo(f"\n{col} statistics:")
            click.echo(f"Minimum: {new_targets_df[col].min()}")
            click.echo(f"Maximum: {new_targets_df[col].max()}")
            click.echo(f"Average: {new_targets_df[col].mean():.2f}")


@click.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the input CSV file containing the data",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path where the targets CSV will be saved/updated",
)
@click.option(
    "--target-columns",
    type=str,
    required=True,
    help="Comma-separated list of target column names to extract",
)
def extract_targets_cli(
    input_csv: Path,
    output_path: Path,
    target_columns: str,
) -> None:
    """Extract target variables from CSV and append to existing targets file."""
    target_cols = [col.strip() for col in target_columns.split(",")]

    click.echo(f"Extracting target variables: {', '.join(target_cols)}")
    extract_targets_from_csv(input_csv, output_path, target_cols)
