from typing import Dict, List

from pydantic_settings import BaseSettings, SettingsConfigDict


class PreprocessingSettings(BaseSettings):
    operating_conditions_initial_column_set: List[str] = []
    pretraining_dataset_mode: str = "unpaired"


class ModelSettings(BaseSettings):
    pad_token_idx: int = 0
    unknown_token_idx: int = 1
    unknown_numerical_value: float = -101.0
    single_split_validation_size: float = 0.05
    embedded_features_prefix: str = "val_"

    model_config = SettingsConfigDict(env_prefix="BEEP_SETTINGS_MODEL_")


class DataHandlerSettings(BaseSettings):
    excluded_devices: List[str] = []
    device_design_data_columns: Dict[str, List[str]] = {"device": ["device"]}
    categorical_device_design_data_columns: Dict[str, List[str]] = {"device": []}
    columns_mapping: Dict[str, str] = {}
    seed_for_operating_conditions_epsilon: int = 10

    model_config = SettingsConfigDict(env_prefix="BEEP_SETTINGS_DATA_HANDLER_")


PREPROCESSING_SETTINGS = PreprocessingSettings()
MODEL_SETTINGS = ModelSettings()
DATA_HANDLER_SETTINGS = DataHandlerSettings()
