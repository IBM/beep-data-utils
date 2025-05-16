"""Dataset generator."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..configuration import MODEL_SETTINGS, PREPROCESSING_SETTINGS
from .preprocessor import BeepDataPreprocessor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logging.basicConfig()


class BeepDatasetGenerator:
    """Beep dataset generator."""

    def __init__(
        self,
        dataset_paths: List[Path],
        sampling_frequencies: Dict[str, Optional[str]],
        target_variables: List[str],
        drop_cycle_zero: bool = False,
        cycle_based_averaging: bool = False,
    ):
        """Initialize BeepDatasetGenerator.

        Args:
            dataset_paths: List of paths to the datasets
            sampling_frequencies: List of sampling frequencies for each dataset
            drop_cycle_zero: Whether to drop cycle zero data
        """
        self.preprocessor = BeepDataPreprocessor(
            testing_data_paths=dataset_paths,
            sampling_frequencies=sampling_frequencies,
            target_variables=target_variables,
            drop_cycle_zero=drop_cycle_zero,
            cycle_based_averaging=cycle_based_averaging,
        )

        logger.info(f"Data loaded for {len(self.preprocessor.experiments)} experiments.")

    def prepare_dataset_for_pretraining_and_fine_tuning(
        self,
        uniformed_dataset_path: Path,
        mask_nan_values: bool,
        unpaired_dataset_output_path: Optional[Path],
        sampling_frequency: Optional[str],
        **kwargs,
    ) -> None:
        """Prepare dataset.

        Filters apply only to operating conditions.
        Filtering is based on percentile unless absolute thresholds are specified in the absolute_thresholds_path JSON file.

        Args:
            uniformed_dataset_path: path where to store the data set.
            mask_nan_values: whether to mask or drop nan values.
            unpaired_dataset_output_path: optional path where to store the unpaired dataset.
            sampling_frequency: sampling frequency for the generated instances.
        """
        initial_devices = [experiment.device.name for experiment in self.preprocessor.experiments]

        self.log_preprocessor_state()

        logger.info("Initialize column list.")
        self.preprocessor.initialize_column_list()
        initial_overlapping_columns = (
            PREPROCESSING_SETTINGS.operating_conditions_initial_column_set.copy()
        )
        logger.info("Initialize column list completed.")

        self.log_preprocessor_state()

        if mask_nan_values:
            logger.info("Masking NaN values using the unknown numerical value.")
            self.preprocessor.mask_nan_values()
            logger.info("Masking NaN values using the unknown numerical value completed.")
        else:
            logger.info("Removing NaN values from the dataset.")
            self.preprocessor.remove_nans()
            logger.info("Removing NaN values from the dataset completed.")

        self.log_preprocessor_state()

        logger.info(f"Generating dataset from {len(self.preprocessor.experiments)} experiments...")

        (
            minimum_number_of_tokens,
            _,
        ) = self.preprocessor.generate_dataset(
            output_path=uniformed_dataset_path,
            mask_nan_values=mask_nan_values,
        )
        logger.info("Dataset generated.")

        path_to_dataset_for_pretraining = uniformed_dataset_path
        if PREPROCESSING_SETTINGS.pretraining_dataset_mode == "unpaired":
            if unpaired_dataset_output_path is None:
                raise ValueError(
                    "Pretraining dataset mode unpaired requires an unpaired_dataset_output_path."
                )
            path_to_dataset_for_pretraining = unpaired_dataset_output_path

        generated_dataset_dataframe = pd.read_csv(path_to_dataset_for_pretraining)

        max_token_length = len([
            column for column in generated_dataset_dataframe.columns if column.startswith("token")
        ])

        final_devices = list(
            set([index.split("/")[0] for index in generated_dataset_dataframe["index"]])
        )
        final_columns = list(generated_dataset_dataframe.columns)

        summary = {
            "number of included devices": len(final_devices),
            "number of initial devices": len(initial_devices),
            "number of included columns": len([
                column for column in initial_overlapping_columns if column in final_columns
            ]),
            "number of initial columns": len(initial_overlapping_columns),
            "number of total columns": len(final_columns),
            "number of unknown numerical value": int(
                generated_dataset_dataframe[
                    generated_dataset_dataframe == MODEL_SETTINGS.unknown_numerical_value
                ]
                .notnull()
                .sum()
                .sum()
            ),
            "number of instances": len(generated_dataset_dataframe),
            "number of tokens": minimum_number_of_tokens,
            "max token length": max_token_length,
            "missing devices": [
                device for device in initial_devices if device not in final_devices
            ],
            "missing columns": [
                column for column in initial_overlapping_columns if column not in final_columns
            ],
            "included devices": final_devices,
            "final columns": final_columns,
            "token2id": {
                k: v
                for k, v in sorted(
                    self.preprocessor.experiments.token2id.items(), key=lambda item: item[1]
                )
            },
        }

        summary_path = str(uniformed_dataset_path).replace(".csv", "_summary.json")
        with open(summary_path, "w") as out:
            json.dump(summary, out, indent=4)

    def log_preprocessor_state(self, log_string_prefix: Optional[str] = None) -> None:
        if log_string_prefix:
            logger.info(log_string_prefix)
        logger.info(
            f"Current columns: {self.preprocessor.experiments._overlapping_operating_conditions_columns}"
        )
        logger.info(
            f"Number of experiments: {sum([1 if len(experiment.operating_conditions) > 1 else 0 for experiment in self.preprocessor.experiments])}"
        )
        logger.info(
            f"Count of all rows of all experiments: {sum([len(experiment.operating_conditions) for experiment in self.preprocessor.experiments])}"
        )
        logger.info(
            f"Number of nans: {sum([experiment.operating_conditions.isna().sum().sum() for experiment in self.preprocessor.experiments])}"
        )
