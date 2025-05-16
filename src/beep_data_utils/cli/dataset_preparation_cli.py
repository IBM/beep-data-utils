"""Dataset preparation with multiple input files support."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import click

from ..data.beep_dataset_preparation import BeepDatasetGenerator

logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    "testing_data_paths", nargs=-1, type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--sampling_frequencies",
    required=False,
    help="Comma-separated list of sampling frequencies, one per input file. Use 'none' for no sampling. "
    "E.g., time dependant: 1s , 0.1s, or calculating statistical measurements over a full cycle: cycle",
    default="cycle",
)
@click.option(
    "--target_variables",
    required=False,
    help="Comma-separated list of variables to be marked as targets",
    default="",
)
@click.option(
    "--uniformed_dataset_path",
    required=True,
    type=click.Path(path_type=Path),
)
@click.option("--mask_nan_values", is_flag=True, show_default=True, default=False)
@click.option("--drop_cycle_zero", is_flag=True, show_default=True, default=False)
@click.option(
    "--cycle_based_averaging",
    is_flag=True,
    show_default=True,
    default=False,
    help="If the averaging should be done on a cycle basis",
)
@click.option(
    "--unpaired_dataset_output_path",
    required=False,
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--sampling_frequency",
    required=False,
    type=str,
    default="1s",
)
def prepare_dataset_for_pretraining_and_fine_tuning(
    testing_data_paths: List[Path],
    sampling_frequencies: str,
    target_variables: str,
    uniformed_dataset_path: Path,
    drop_cycle_zero: bool,
    cycle_based_averaging: bool,
    mask_nan_values: bool,
    unpaired_dataset_output_path: Optional[Path],
    sampling_frequency: Optional[str],
) -> None:
    """Prepare dataset.

    Filters apply only to testing data.

    Args:
        testing_data_paths: List of paths to testing data files.
        sampling_frequencies: Comma-separated list of sampling frequencies if you different ones for certain datasets.
        target_variables: Comma-separated list of variables to be marked as targets.
        uniformed_dataset_path: Path to store the dataset.
        limits_file_path: Optional path to feature limits.
        drop_cycle_zero: Whether to drop cycle zero data.
        cycle_based_averaging: If the averaging/downsampling should be done on a cycle basis.
        mask_nan_values: Whether to mask NaN values.
        unpaired_dataset_output_path: Optional path for unpaired dataset.
        sampling_frequency: Overall sampling frequency for the generated instances.
    """
    # Create list of Optional[str] for frequencies
    freq_list: List[Optional[str]] = []
    if sampling_frequencies:
        freq_list = [
            None if f.lower() == "none" else f.strip() for f in sampling_frequencies.split(",")
        ]

    if len(freq_list) < len(testing_data_paths):
        freq_list.extend([None] * (len(testing_data_paths) - len(freq_list)))

    sampling_freq_dict: Dict[str, Optional[str]] = {
        str(path): freq for path, freq in zip(testing_data_paths, freq_list)
    }

    target_vars = [var.strip() for var in target_variables.split(",")] if target_variables else []

    generator = BeepDatasetGenerator(
        dataset_paths=testing_data_paths,
        sampling_frequencies=sampling_freq_dict,
        target_variables=target_vars,
        drop_cycle_zero=drop_cycle_zero,
        cycle_based_averaging=cycle_based_averaging,
    )

    if unpaired_dataset_output_path:
        uniformed_dataset_path = unpaired_dataset_output_path
    generator.prepare_dataset_for_pretraining_and_fine_tuning(
        uniformed_dataset_path,
        mask_nan_values,
        unpaired_dataset_output_path,
        sampling_frequency,
    )
