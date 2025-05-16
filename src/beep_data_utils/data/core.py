"""Data modules."""

import logging
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from numpy.typing import NDArray
from pandas.core.generic import NDFrame

from ..configuration import DATA_HANDLER_SETTINGS, MODEL_SETTINGS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logging.basicConfig()


class Element:
    """Abstract base class."""

    def __init__(self, data: pd.Series) -> None:
        """Initialize an Element.

        Args:
            data: data of the element.
        """
        self.data = data

    def to_pandas(self) -> NDFrame:
        """Return data as pandas series.

        Returns:
            a pandas series.
        """
        return self.data

    def to_numpy(self) -> NDArray[Any]:
        """Return data as numpy array.

        Returns:
            numpy array with the data.
        """
        return self.to_pandas().to_numpy()

    def __getitem__(self, attr: Any) -> Any:
        """Get item from underlying pandas series.

        Args:
            attr: attribute to return.

        Returns:
            item matching the attribute.
        """
        return self.data[attr]

    def iloc(self, index: int) -> Any:
        """Get item from underlying pandas series via integer position.

        Args:
            index: position in the series.

        Returns:
            item at the requested position.
        """
        return self.data.iloc[index]


class Device(Element):
    """Device class."""

    design_data_token_ids: List[int]

    def __init__(
        self,
        device_data: pd.Series,
    ) -> None:
        """Initialize a device.

        Args:
            device_data: data for the device.
        """
        super().__init__(device_data)

        self.name = device_data["device"]

        self.data.drop(["device"], inplace=True)

    def device_information_to_device_feature_list(self) -> List[str]:
        """Convert device information to a list of singular features that characterize the device.

        Returns:
            List of strings to represent the device design information.
        """
        # TODO: consider normalizing further the parameters (currently we set to lowercase)
        tokens = [
            f"{parameter_name.lower()}_{str(parameter).lower()}"
            for parameter_name, parameter in self.categorical_device_design_features_to_pandas()
            .to_dict()
            .items()
        ]
        return tokens

    def tokenize_design_data(self, token2id: Dict[str, int]):
        """Tokenize device info using the provided token2id mapper.

        Args:
            token2id: token to id mapper.
        """
        tokens = self.device_information_to_device_feature_list()

        self.design_data_token_ids = [
            token2id.get(token, MODEL_SETTINGS.unknown_token_idx) for token in tokens
        ]

    def categorical_device_design_features_to_pandas(self) -> pd.Series:
        """Return categorical features of a device as pandas data frame.

        Returns:
            categorical features data frame.
        """
        existing_categorical_device_columns = [
            column
            for column in DATA_HANDLER_SETTINGS.categorical_device_design_data_columns["device"]
            if column in self.data.keys()
        ]
        return self.data[existing_categorical_device_columns]

    def numerical_device_design_features_to_pandas(self) -> pd.Series:
        """Return numerical features of a device as pandas data frame.

        Returns:
            numerical features data frame.
        """
        return self.data[
            [
                column
                for column in self.data.keys()
                if column
                not in DATA_HANDLER_SETTINGS.categorical_device_design_data_columns["device"]
            ]
        ]


class Experiment:
    def __init__(
        self,
        operating_conditions: pd.DataFrame,
        device: Device,
    ) -> None:
        """Initialize an experiment.

        Args:
            operating_conditions: the operating conditions of the experiment.
            device: device of the experiment.
        """
        self.device = device
        self.operating_conditions = operating_conditions

    def get_experiment_components(self):
        """Get all the components of the experiment.

        Returns:
            A dictionary with all the components of the experiment.
        """
        class_dict = {
            "device": self.device,
            "operating_conditions": self.operating_conditions,
        }
        return class_dict

    def operating_conditions_to_pandas(self) -> pd.DataFrame:
        """Return operating conditions data as data frame.

        Returns:
            data frame with the operating conditions data.
        """
        return self.operating_conditions

    def operating_conditions_to_numpy(self) -> NDArray[Any]:
        """Return operating conditions data as numpy array.

        Returns:
            numpy array with the operating conditions data.
        """
        return self.operating_conditions_to_pandas().to_numpy()

    def device_to_pandas(self) -> pd.Series:
        """Return device data as data frame.

        Returns:
            data frame with the device data.
        """
        return self.device.to_pandas()

    def device_to_numpy(self) -> NDArray[Any]:
        """Return device data as numpy array.

        Returns:
            numpy array with the device data.
        """
        return self.device.to_numpy()

    def get_device(self) -> Device:
        """Get the device of the experiment.

        Returns:
            the experiment's device.
        """
        return self.device


class ExperimentSet:
    def __init__(
        self,
        experiments: List[Experiment],
        do_tokenize_device_design_data: bool = False,
        vocabulary: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize a set of experiments.

        Args:
            experiments: A list of experiments.
            do_tokenize_device_design_data: option to tokenize. Default to False.
            vocabulary: vocabulary to be used for the device features.
        """
        self.experiments = experiments
        self.do_tokenize_device_design_data = do_tokenize_device_design_data

        self.init(vocabulary)

        if do_tokenize_device_design_data:
            self.tokenize_device_design_data()

    def number_of_tokens(self) -> int:
        """Returns the number of tokens used for device representation.

        Returns:
            number of tokens.
        """
        return len(self.token2id) + 2  # numerical token + unknown numerical token

    def init(self, vocabulary: Optional[Dict[str, int]] = None) -> None:
        """Initialize object's parameters.

        Args:
            vocabulary: vocabulary to be used for the device features.
        """

        self._overlapping_operating_conditions_columns = sorted(
            set.intersection(*[
                set(experiment.operating_conditions_to_pandas().columns.to_list())
                for experiment in self.experiments
            ])
        )
        self._all_operating_conditions_columns = sorted(
            set(self._overlapping_operating_conditions_columns)
        )
        self._all_operating_conditions_columns = sorted(
            set.union(*[
                set(experiment.operating_conditions_to_pandas().columns.to_list())
                for experiment in self.experiments
            ])
        )

        if vocabulary:
            self.token2id = vocabulary
        else:
            self.token2id = self.create_token2id_mapper()

    def append(self, experiments: List[Experiment]) -> None:
        """Add experiments in the object.

        Args:
            experiments: List of new experiments to be added.
        """
        self.experiments += experiments

        self.init()

    def tokenize_device_design_data(self) -> None:
        """Tokenize device info of all the devices."""
        self.device_design_data_tokenized = True

        for experiment in self.experiments:
            experiment.device.tokenize_design_data(self.token2id)

    def create_token2id_mapper(self) -> Dict[str, int]:
        """Create token2id mapper based on all the devices of the experiments.

        Returns:
            Dictionary of device information to id mapping.
        """
        tokens = []
        for experiment in self.experiments:
            tokens += experiment.device.device_information_to_device_feature_list()
        unique_tokens = list(set(tokens))

        unique_tokens.sort()

        token2id = {token: i for i, token in enumerate(unique_tokens, start=2)}
        token2id["PAD_TOKEN"] = MODEL_SETTINGS.pad_token_idx
        token2id["UNK_TOKEN"] = MODEL_SETTINGS.unknown_token_idx

        return token2id

    def __getitem__(self, index: int) -> Experiment:
        """Get experiments by index.

        Args:
            index: index of the experiment.

        Returns:
            an experiment.
        """
        return self.experiments[index]

    def __len__(self) -> int:
        """Get the number of experiments.

        Returns:
            Number of experiments
        """
        return len(self.experiments)

    def __iter__(self) -> Iterator[Experiment]:
        for i in range(len(self)):
            yield self[i]


class DeviceDataIndex:
    def __init__(
        self, device_identifier: str, start_time: float, delta_time: Optional[float] = None
    ) -> None:
        """Initialize device data index.

        Args:
            device_identifier: identifier of the device.
            start_time: start time for the operating conditions.
            delta_time: delta time.
        """
        self.device_identifier = device_identifier
        self.start_time = start_time
        self.delta_time = delta_time
        if self.delta_time is None:
            self.end_time = None
        else:
            self.end_time = self.start_time + self.delta_time

    @staticmethod
    def from_string_index(index: str) -> "DeviceDataIndex":
        """Instantiate a DeviceDataIndex from a string index.

        Args:
            index: string index.

        Returns:
            device data index object.
        """
        splitted_index = index.split("/")

        device_identifier = splitted_index[0]
        start_time = float(splitted_index[1])
        delta_time = None
        if len(splitted_index) == 3:
            delta_time = float(splitted_index[2])

        return DeviceDataIndex(
            device_identifier=device_identifier,
            start_time=start_time,
            delta_time=delta_time,
        )

    def __str__(self) -> str:
        """DeviceDataIndex as string.

        Returns:
            stringified device data index.
        """
        if self.delta_time is None:
            return f"{self.device_identifier}/{self.start_time}"
        return f"{self.device_identifier}/{self.start_time}/{self.delta_time}"
