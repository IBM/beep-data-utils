from pathlib import Path
from typing import List, Set, Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split


def get_unique_cell_keys(data_path: Path) -> List[str]:
    """Extract unique cell keys from the CSV file."""
    df = pd.read_csv(data_path)
    return sorted(df["cell_key"].unique())


def read_existing_domain_file(file_path: Path) -> Set[str]:
    """Read existing cell keys from a file if it exists."""
    if not file_path.exists():
        return set()

    with open(file_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def write_domain_file(cells: List[str], output_path: Path, append: bool = False) -> None:
    """Write cell keys to a domain file."""
    # First read existing content if appending
    existing_content = []
    if append and output_path.exists():
        with open(output_path, "r") as f:
            existing_content = [line.strip() for line in f if line.strip()]

    # Combine existing and new content
    all_cells = existing_content + cells if append else cells

    # Write all content
    with open(output_path, "w") as f:
        for cell in all_cells:
            f.write(f"{cell}\n")


def split_domains(
    cell_keys: List[str], test_size: float = 0.1, random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """Split cell keys into in-domain and out-of-domain sets."""
    in_domain, out_domain = train_test_split(
        cell_keys, test_size=test_size, random_state=random_state
    )
    return sorted(in_domain), sorted(out_domain)


@click.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input CSV file containing battery cell data",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Directory where in_domain.txt and out_of_domain.txt will be saved",
)
@click.option(
    "--test-size",
    type=float,
    default=0.1,
    help="Fraction of cells to use for out-of-domain testing (default: 0.1)",
)
@click.option(
    "--random-state", type=int, default=42, help="Random seed for reproducibility (default: 42)"
)
@click.option(
    "--append/--no-append",
    default=False,
    help="Append to existing files instead of overwriting (default: False)",
)
def generate_domain_splits(
    input_csv: str, output_dir: str, test_size: float, random_state: int, append: bool
) -> None:
    """Generate in-domain and out-of-domain splits from battery cell data CSV."""
    data_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    in_domain_path = output_path / "in_domain.txt"
    out_domain_path = output_path / "out_of_domain.txt"

    # Get existing cells if appending
    existing_in_domain = read_existing_domain_file(in_domain_path) if append else set()
    existing_out_domain = read_existing_domain_file(out_domain_path) if append else set()

    all_cell_keys = get_unique_cell_keys(data_path)

    # Filter out existing cells if appending
    if append:
        cell_keys = [
            key
            for key in all_cell_keys
            if key not in existing_in_domain and key not in existing_out_domain
        ]
        if not cell_keys:
            click.echo("All cells from the CSV already exist in domain files. Nothing to do.")
            return
        click.echo(f"Found {len(cell_keys)} new unique cells")
    else:
        cell_keys = all_cell_keys
        click.echo(f"Found {len(cell_keys)} unique cells")

    # Perform the split
    click.echo(f"Splitting cells with test_size={test_size}")
    in_domain, out_domain = split_domains(cell_keys, test_size, random_state)

    # Write output files
    write_domain_file(in_domain, in_domain_path, append)
    write_domain_file(out_domain, out_domain_path, append)

    # Get final counts
    final_in_domain = read_existing_domain_file(in_domain_path)
    final_out_domain = read_existing_domain_file(out_domain_path)

    # Display summary
    click.echo("\nSplit Summary:")
    if append:
        click.echo(f"New in-domain cells: {len(in_domain)}")
        click.echo(f"New out-domain cells: {len(out_domain)}")
    click.echo(f"Total in-domain cells: {len(final_in_domain)}")
    click.echo(f"Total out-domain cells: {len(final_out_domain)}")

    click.echo("\nFiles written to:")
    click.echo(f"- {in_domain_path}")
    click.echo(f"- {out_domain_path}")
