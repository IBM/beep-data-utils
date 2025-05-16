__copyright__ = """

"""

import random

import numpy as np
import torch
from beep_data_utils.modeling.heads.hf_heads import (
    LinearRegressor,
    LinearRegressorConfig,
    LSTMRegressor,
    LSTMRegressorConfig,
)


def lstm_regressor(seed=10):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    batch_size = torch.randint(low=1, high=127, size=(1,)).item()

    embedding_dim = torch.randint(low=1, high=127, size=(1,)).item()
    lstm_dim = torch.randint(low=1, high=127, size=(1,)).item()
    seq_len = torch.randint(low=1, high=127, size=(1,)).item()
    regressed_channel = torch.randint(low=1, high=127, size=(1,)).item()
    num_layer = torch.randint(low=1, high=12, size=(1,)).item()

    modelConfig = LSTMRegressorConfig(
        num_input_feature=embedding_dim,
        hidden_size=lstm_dim,
        num_output_feature=regressed_channel,
        num_layers=num_layer,
    )

    X = torch.randn(size=(batch_size, seq_len, embedding_dim))
    regressor = LSTMRegressor(modelConfig)
    pred = regressor(X)
    assert pred["logits"].shape == (batch_size, regressed_channel), "Error in the code"


def linear_regressor(seed=10):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    batch_size = torch.randint(low=1, high=127, size=(1,)).item()

    embedding_dim = torch.randint(low=1, high=127, size=(1,)).item()
    seq_len = torch.randint(low=1, high=127, size=(1,)).item()
    regressed_channel = torch.randint(low=1, high=127, size=(1,)).item()
    num_layer = torch.randint(low=1, high=12, size=(1,)).item()

    modelConfig = LinearRegressorConfig(
        num_input_feature=embedding_dim,
        seq_len=seq_len,
        num_output_feature=regressed_channel,
        num_layers=num_layer,
    )
    X = torch.randn(size=(batch_size, seq_len, embedding_dim))
    regressor = LinearRegressor(modelConfig)
    regressor.save_pretrained("training_output_dir/model_save")
    pred = regressor.forward(X)
    assert pred["logits"].shape == (batch_size, regressed_channel), "Error in the code"


def test_lstm_regressor():
    for i in range(10):
        lstm_regressor(i)


def test_linear_regressor():
    for i in range(10):
        linear_regressor(i)
