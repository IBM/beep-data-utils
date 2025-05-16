from pathlib import Path

import importlib_resources
import numpy as np
import polars as pl
import pytest
import torch
from beep_data_utils.modeling.heads.hf_utils import SeqDataset
from transformers import default_data_collator


def get_root_dir():
    return Path(__file__).parent.parent


@pytest.fixture
def test_dataframe_unnormalized():
    """Create a sample dataframe for testing"""
    return get_dataframe_fn().with_columns(pl.col("^val_.*$") * 5 + 1)


# Fixture should not be called directly, therefore we wrapped the function
@pytest.fixture
def test_dataframe():
    return get_dataframe_fn()


def get_dataframe_fn():
    """Create a sample dataframe for testing"""
    with importlib_resources.as_file(
        importlib_resources.files("beep_data_utils") / "resources/beep-train.csv"
    ) as data_path:
        return pl.read_csv(data_path).with_columns(pl.col("index").alias("battery"))


def get_input_features():
    return [f"val_{i}" for i in range(54)] + [
        "charge_policy_Q1",
        "charge_policy_Q2",
        "charge_policy_Q3",
        "charge_policy_Q4",
    ]


def get_target_features():
    return ["charge_capacity"]


def test_seqdataset_initialization(test_dataframe):
    """Test initialization of SeqDataset"""
    dataset = SeqDataset(
        dataset=test_dataframe,
        input_features=get_input_features(),
        target_features=get_target_features(),
    )
    assert len(dataset) > 0
    assert dataset.cutoff == 100
    assert dataset.col_id == "battery"
    assert dataset.time_unit == "cycle_number"


def test_seqdataset_initialization_unnormed_g(test_dataframe_unnormalized):
    """Test initialization of SeqDataset"""
    ds = SeqDataset(
        dataset=test_dataframe_unnormalized,
        input_features=get_input_features(),
        target_features=get_target_features(),
        normalize_columns=False,
        cutoff=56,
    )
    print()
    batched_input = default_data_collator([ds[i] for i in range(0, len(ds) // 2, 2)])
    X = batched_input["input_ids"]
    assert not torch.isclose(torch.mean(X[:, :, :-1]), torch.tensor(0.0, dtype=torch.float))


def test_seqdataset_get_item_returns_dict_with_correct_shape(test_dataframe_unnormalized):
    """Test initialization of SeqDataset"""
    for cutoff in range(80, 105):
        dataset = SeqDataset(
            dataset=test_dataframe_unnormalized,
            input_features=get_input_features(),
            target_features=get_target_features(),
            normalize_columns=False,
            cutoff=cutoff,
        )
        i = np.random.randint(0, len(dataset))
        input = dataset[i]
        assert "input_ids" in input.keys()
        assert "labels" in input.keys()
        X = input["input_ids"]
        y = input["labels"]

        # We have a cutoff sequence times the target_cycle_number + number of input
        assert X.shape == (cutoff, 1 + len(get_input_features()))
        # We have predict at the target cycle number the target features
        assert y.shape[0] == 1


def test_seqdataset_gets_normalized_values_mean_zero(test_dataframe_unnormalized):
    """Test initialization of SeqDataset"""
    dataset = SeqDataset(
        dataset=test_dataframe_unnormalized,
        input_features=get_input_features(),
        target_features=get_target_features(),
        normalize_columns=True,
        cutoff=99,
    )
    input_data = dataset[1]
    batched_input = default_data_collator([dataset[i] for i in range(0, len(dataset) // 2)])

    assert "labels" in input_data.keys() and "labels" in batched_input.keys()
    assert "input_ids" in input_data.keys() and "input_ids" in batched_input.keys()
    X = batched_input["input_ids"]

    # Mean is 0 after normalization
    # We remove the target cycle number
    mean_of_features = torch.mean(X[:, :, :-1])
    assert torch.isclose(mean_of_features, torch.tensor(0.0), atol=1e-2)
