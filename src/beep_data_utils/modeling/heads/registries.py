"""Registries definitions for head objects."""

from typing import Dict

from transformers import PretrainedConfig, PreTrainedModel

from .hf_heads import (
    LinearRegressor,
    LinearRegressorConfig,
    LSTMRegressor,
    LSTMRegressorConfig,
)

HF_HEAD: Dict[str, PreTrainedModel] = {
    "LSTM": LSTMRegressor,
    "linear": LinearRegressor,
}
HF_HEAD_CONFIG: Dict[str, PretrainedConfig] = {
    "LSTM": LSTMRegressorConfig,
    "linear": LinearRegressorConfig,
}
