import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..configuration import MODEL_SETTINGS, PREPROCESSING_SETTINGS
from .core import Experiment, ExperimentSet
from .handler import BeepDataHandler
from .handler_cycle_based_averaging import BeepDataHandlerCycle

logger = logging.getLogger(__name__)

EXPERIMENT_TO_DATA_FRAME_FN_MAPPING: Dict[str, Callable[[Experiment], pd.DataFrame]] = {
    "operating_conditions": lambda experiment: experiment.operating_conditions_to_pandas(),
}

EXPERIMENT_TO_COLUMNS_MAPPING: Dict[str, Callable[[ExperimentSet], List[str]]] = {
    "operating_conditions": lambda experiment_set: experiment_set._overlapping_operating_conditions_columns,
}

CATEGORICAL_COLUMNS_MAPPING: Dict[str, List[str]] = {
    "operating_conditions": [],
}

NUMERICAL_TYPES = (int, float)


class BeepDataPreprocessor:
    def __init__(
        self,
        testing_data_paths: List[Path],
        sampling_frequencies: Dict[str, Optional[str]],
        target_variables: List[str],
        drop_cycle_zero: bool = False,
        cycle_based_averaging: bool = False,
    ) -> None:
        """Preprocessor.

        Args:
            testing_data_paths: List of paths to the CSVs.
            sampling_frequencies: Dictionary mapping data paths to their sampling frequencies.
                If a path is not in the dict or mapped to None, no resampling is done.
            drop_cycle_zero: whether to drop cycle zero data
        """
        handler = BeepDataHandler()
        cycle_handler = BeepDataHandlerCycle()
        all_experiments = []
        self.target_variables = target_variables

        # Process each dataset with its specific sampling frequency
        if cycle_based_averaging:
            for data_path in testing_data_paths:
                sampling_freq = sampling_frequencies.get(str(data_path))
                exps = cycle_handler.read_experiments(
                    testing_data_path=data_path,
                    target_variables=target_variables,
                    drop_cycle_zero=drop_cycle_zero,
                    sampling_frequency=sampling_freq,
                )
                all_experiments.extend(exps.experiments)
        else:
            for data_path in testing_data_paths:
                sampling_freq = sampling_frequencies.get(str(data_path))
                exps = handler.read_experiments(
                    testing_data_path=data_path,
                    target_variables=target_variables,
                    drop_cycle_zero=drop_cycle_zero,
                    sampling_frequency=sampling_freq,
                )
                all_experiments.extend(exps.experiments)

        # Create a single ExperimentSet from all experiments
        self.experiments = ExperimentSet(all_experiments)
        self.not_processable_features = [["cell_key", "batch_number", "time"]]

    def process_experiment_set_data_frame(
        self,
        data_frame_processing_fn: Callable,
        processing_function_input: Optional[Any] = None,
        experiment_to_data_frame_fn_name: str = "operating_conditions",
    ):
        """Process in-place experiment set using a data frame.

        Args:
            feature_name: feature name.
            data_frame_processing_fn: function for data frame processing for specific column.
            experiment_to_data_frame_fn_name: key to get the callable used to select the data frame for filtering.
                Defaults to using operating conditions.
        """
        experiment_to_data_frame_fn = self.get_experiment_to_data_frame_fn(
            experiment_to_data_frame_fn_name
        )
        experiment_list: List[Experiment] = []
        for experiment in self.experiments:
            try:
                experiment_kwargs = experiment.get_experiment_components()

                data_frame = experiment_to_data_frame_fn(experiment)

                if processing_function_input:
                    experiment_kwargs[experiment_to_data_frame_fn_name] = data_frame_processing_fn(
                        data_frame, processing_function_input
                    )
                else:
                    experiment_kwargs[experiment_to_data_frame_fn_name] = data_frame_processing_fn(
                        data_frame
                    )

                experiment_list.append(Experiment(**experiment_kwargs))

            except Exception:
                error_details = (
                    f"error encountered when processing data frame {experiment_to_data_frame_fn_name} for experiment "
                    f"{experiment} (kwargs={experiment.__dict__}) using function {data_frame_processing_fn} (arguments={processing_function_input})."
                )
                logger.warning(
                    f"experiment for device {experiment.device.name} could not be processed (data frame {experiment_to_data_frame_fn_name})! It will be discarded. Reason: {error_details}"
                )

    def add_columns_with_nans_and_sort(
        self, data_frame: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Add uninitialized columns to a data frame.

        Args:
            data_frame: a data frame.
            columns: list of columns to be added in the data frame.

        Returns:
            the data frame with added columns.
        """
        columns_to_be_added = [column for column in columns if column not in data_frame]
        data_frame[columns_to_be_added] = np.NAN
        df_columns = list(data_frame.columns)
        df_columns.sort()
        data_frame = data_frame[df_columns]
        return data_frame

    def initialize_column_list(
        self,
        experiment_to_data_frame_fn_name="operating_conditions",
    ) -> None:
        """Remove all columns which are not overlapping.

        Args:
            experiment_to_data_frame_fn_name: key to get the callable used to select the data frame for filtering.
                Defaults to "operating_conditions".
        """
        if experiment_to_data_frame_fn_name != "operating_conditions":
            raise ValueError("set_column_list is supported only for operating conditions.")

        if PREPROCESSING_SETTINGS.operating_conditions_initial_column_set:
            columns_of_interest = PREPROCESSING_SETTINGS.operating_conditions_initial_column_set
        else:
            columns_of_interest = EXPERIMENT_TO_COLUMNS_MAPPING[experiment_to_data_frame_fn_name](
                self.experiments
            )

        experiment_to_data_frame_fn = self.get_experiment_to_data_frame_fn(
            experiment_to_data_frame_fn_name
        )
        set_of_all_columns: Set[str] = set()
        for experiment in tqdm(self.experiments, desc="compute union"):
            data_frame = experiment_to_data_frame_fn(experiment)
            set_of_all_columns = set_of_all_columns.union(set(data_frame.columns))

        columns_to_be_excluded = list(set_of_all_columns - set(columns_of_interest))

        if len(columns_to_be_excluded) > 0:
            self.remove_feature_from_all_experiments(
                columns_to_be_excluded,
                experiment_to_data_frame_fn_name=experiment_to_data_frame_fn_name,
            )

        self.process_experiment_set_data_frame(
            processing_function_input=columns_of_interest,
            data_frame_processing_fn=self.add_columns_with_nans_and_sort,
            experiment_to_data_frame_fn_name=experiment_to_data_frame_fn_name,
        )

    def remove_nans(self, experiment_to_data_frame_fn_name: str = "operating_conditions"):
        """Remove NaN rows.

        Args:
            experiment_to_data_frame_fn_name: key to get the callable used to select the data frame for filtering.
                Defaults to using operating conditions.
        """

        self.process_experiment_set_data_frame(
            data_frame_processing_fn=self._remove_nans_processing_fn,
            processing_function_input=experiment_to_data_frame_fn_name,
            experiment_to_data_frame_fn_name=experiment_to_data_frame_fn_name,
        )

    def _remove_nans_processing_fn(
        self,
        data_frame_to_process: pd.DataFrame,
        experiment_to_data_frame_fn_name: str,
    ) -> pd.DataFrame:
        """Remove NaNs.

        Args:
            data_frame_to_process: data frame to process.
            experiment_to_data_frame_fn_name: key to get the callable used to select the data frame for filtering.
                Defaults to using operating conditions.
        """
        for feature_name in data_frame_to_process.columns:
            is_numerical_feature = self.is_numerical_feature(
                feature_name=feature_name,
                experiment_to_data_frame_fn_name=experiment_to_data_frame_fn_name,
                numerical_fraction=0.9,
            )
            if is_numerical_feature:
                feature_column_with_non_numeric_replaced_by_nans = pd.to_numeric(
                    data_frame_to_process[feature_name], errors="coerce"
                )
                data_frame_to_process[feature_name] = (
                    feature_column_with_non_numeric_replaced_by_nans
                )

        data_frame_to_process.dropna(inplace=True)

        return data_frame_to_process

    def mask_nan_values(self, experiment_to_data_frame_fn_name: str = "operating_conditions"):
        """Remove NaN rows.

        Args:
            experiment_to_data_frame_fn_name: key to get the callable used to select the data frame for filtering.
                Defaults to using operating conditions.
        """

        self.process_experiment_set_data_frame(
            data_frame_processing_fn=lambda data_frame: data_frame.fillna(
                MODEL_SETTINGS.unknown_numerical_value
            ),
            experiment_to_data_frame_fn_name=experiment_to_data_frame_fn_name,
        )

    def remove_feature_from_all_experiments(
        self,
        feature_names_to_drop: Union[str, List[str]],
        experiment_to_data_frame_fn_name: str,
    ) -> None:
        """Remove a feature from all experiments.

        Args:
            feature_name: feature name.
            experiment_to_data_frame_fn_name: key to get the callable used to select the data frame for filtering.
                Defaults to using operating conditions.
        """
        self.process_experiment_set_data_frame(
            processing_function_input=feature_names_to_drop,
            data_frame_processing_fn=lambda data_frame, feature_name: data_frame.drop(
                feature_name, axis=1, errors="ignore"
            ),
            experiment_to_data_frame_fn_name=experiment_to_data_frame_fn_name,
        )

    def get_values_from_experiments(
        self,
        feature_name: str,
        experiment_to_data_frame_fn_name: str = "operating_conditions",
        keep_only_valid_numerical_values: bool = False,
    ) -> List[Any]:
        """Get values from all experiments for a given feature.

        Args:
            feature_name: feature name.
            experiment_to_data_frame_fn_name: key to get the callable used to select the data frame for filtering.
                Defaults to using operating conditions.
            keep_only_valid_numerical_values: whether to keep only valid numerical values.

        Returns:
            list of values.
        """
        experiment_to_data_frame_fn = self.get_experiment_to_data_frame_fn(
            experiment_to_data_frame_fn_name
        )
        values = [
            value
            for experiment in self.experiments
            for value in experiment_to_data_frame_fn(experiment)[feature_name]
        ]
        if keep_only_valid_numerical_values:
            return [
                value
                for value in values
                if isinstance(value, NUMERICAL_TYPES) and not pd.isnull(value)
            ]
        else:
            return values

    def is_numerical_feature(
        self,
        feature_name: str,
        experiment_to_data_frame_fn_name: str,
        numerical_fraction: float = 0.9,
    ) -> bool:
        """Check whether a feature is numerical with a given threshold.

        Args:
            feature_name: feature name.
            experiment_to_data_frame_fn_name: key to get the callable used to select the data frame for filtering.
                Defaults to using operating conditions.
            numerical_fraction: fraction of numerical values to consider a feature numerical. Defaults to 0.9.

        Returns:
            whether a feature is numerical.
        """
        numerical_count = 0
        non_numerical_count = 0
        for value in self.get_values_from_experiments(
            feature_name=feature_name,
            experiment_to_data_frame_fn_name=experiment_to_data_frame_fn_name,
        ):
            if isinstance(value, NUMERICAL_TYPES):
                numerical_count += 1
            else:
                non_numerical_count += 1
        return (
            (numerical_count / float(numerical_count + non_numerical_count)) >= numerical_fraction
            if non_numerical_count > 0
            else True
        )

    @staticmethod
    def get_experiment_to_data_frame_fn(
        experiment_to_data_frame_fn_name: str,
    ) -> Callable[[Experiment], pd.DataFrame]:
        """_summary_

        Args:
            experiment_to_data_frame_fn_name: name of the function associated in the mapping to access a data frame.

        Raises:
            ValueError: in case the name of the function specified does not match any function name in the mapping.

        Returns:
            the function requested.
        """
        if experiment_to_data_frame_fn_name not in EXPERIMENT_TO_DATA_FRAME_FN_MAPPING:
            raise ValueError(
                f"experiment_to_data_frame_fn_name={experiment_to_data_frame_fn_name} not valid! Pick one in {sorted(EXPERIMENT_TO_DATA_FRAME_FN_MAPPING.keys())}."
            )
        experiment_to_data_frame_fn = EXPERIMENT_TO_DATA_FRAME_FN_MAPPING[
            experiment_to_data_frame_fn_name
        ]
        return experiment_to_data_frame_fn

    def experiment_to_instances(
        self,
        experiment: Experiment,
        padded_unit_tokens_length: int,
        padding_token: int = MODEL_SETTINGS.pad_token_idx,
        mask_nan_values: bool = False,
    ) -> pd.DataFrame:
        """Generate instances for a given experiment.

        Args:
            experiment: battery experiment
            padded_unit_tokens_length: padding length for unit tokens
            padding_token: token used for padding
            mask_nan_values: whether to mask NaN values

        Returns:
            DataFrame containing the processed instances
        """
        testing_data = experiment.operating_conditions_to_pandas()

        instances = testing_data.copy()

        tokens = getattr(experiment.device, "design_data_token_ids", [])

        if (padded_unit_tokens_length - len(tokens)) < 0:
            logger.warning(
                f"negative value for padding the tokens: padded_unit_tokens_length is smaller than tokens, {padded_unit_tokens_length}, {len(tokens)}, respectively."
            )
        tokens += [padding_token] * (padded_unit_tokens_length - len(tokens))

        for idx, token in enumerate(tokens):
            instances[f"token_{idx}"] = token

        instances["index"] = [f"{experiment.device.name}/{t}" for t in instances["time"]]

        if mask_nan_values:
            instances.fillna(MODEL_SETTINGS.unknown_numerical_value, inplace=True)

        instances.set_index("index", inplace=True)

        return instances

    def get_padded_device_design_data_tokens_length(self) -> int:
        """Get the padded token length as maximum number of tokens in the dataset.

        Returns:
            maximum number of device design data tokens to be used as padded token length.
        """
        maximum_number_of_tokens = 0
        for experiment in self.experiments:
            experiment.device.tokenize_design_data(self.experiments.token2id)
            maximum_number_of_tokens = max(
                maximum_number_of_tokens,
                len(experiment.device.design_data_token_ids),
            )
        return maximum_number_of_tokens

    def generate_dataset(
        self,
        output_path: Path,
        padding_token: int = MODEL_SETTINGS.pad_token_idx,
        mask_nan_values: bool = False,
    ) -> Tuple[int, int]:
        """Generate processed dataset files.

        Args:
            output_path: path to save output files
            padding_token: token for padding
            mask_nan_values: whether to mask NaN values

        Returns:
            Tuple of (minimum tokens, padded token length)
        """
        padded_device_tokens_length = self.get_padded_device_design_data_tokens_length()

        for i, experiment in tqdm(enumerate(self.experiments), desc="Processing experiments"):
            instances = self.experiment_to_instances(
                experiment,
                padded_device_tokens_length,
                padding_token,
                mask_nan_values,
            )
            mode = "w" if i == 0 else "a"
            header = i == 0

            if len(instances) > 0:
                instances.to_csv(output_path, mode=mode, header=header)

        return (self.experiments.number_of_tokens(), padded_device_tokens_length)
