"""This module implements huggingface API related utilities
Most importantly it provides:
    - A pytorch dataset ready to be used with a dataloader,
        - The output is compliant with the HugggingFace-API
"""

from typing import Dict, List

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import Dataset
from transformers import default_data_collator


class SeqDataset(Dataset):
    """The Sequence dataset, maps the a target timestep to a training sequence and adapt the training to include the target timestamp
    The dataset MUST contain the id column (where the device id is stored) and a time unit
    For example, this can be battery_id and charging cycle number respectively.
    """

    def __init__(
        self,
        dataset: pl.DataFrame,
        input_features: List[str],
        target_features: List[str],
        cutoff: int = 100,
        stratify: bool = False,
        id_column: str = "battery",
        time_unit: str = "cycle_number",
        normalize_columns: bool = True,
        monoton_smooth: List[str] = [],
    ):
        """Init the class,
        This class allows to pair a target and a sequence

        Args:
            dataset (pl.DataFrame): The dataframe on which to operate, each row is a recording of multiple values for a device at a time.
            input_features (List[str]): The list of feature you want to input to your model
            target_features (List[str]): The list of feature you want to predict
            cutoff (int, optional): The cutoff, because we predict from a sequence. Defaults to 100.
            stratify (bool, optional): If all the device should have the same probability to be a training sample irrespective of their life span. Defaults to False.
            id_column (str, optional): The column name in which the devices id are stored. Defaults to "battery".
            time_unit (str, optional): the column name in which the timestep is recorded. Defaults to "cycle_number".
            normalize_columns (bool, optional): If you want to normalize the columns i.e as if the values are drawn from a normal distribution N(0,1)
            monoton_smooth (List[str]): smooth a list target that is monoton decreasing with downwards outliers.
        """
        self.time_unit = time_unit
        self.col_id = id_column
        self.cutoff = cutoff
        self.target_features = target_features
        self.input_features = input_features
        self.stratify = stratify
        assert len(target_features) > 0, "At least 1 feature MUST be provided as target feature"
        assert id_column in dataset.columns and self.time_unit in dataset.columns

        for c in input_features + target_features:
            assert dataset.schema[
                c
            ].is_numeric(), f"Feature {c} is in the dataframe input / target and is not numeric, we won't be able to aggregate per time unit with non-numeric data type"

        assert (
            id_column not in input_features and time_unit not in input_features
        ), "You can't used a column as a group identifier and as an input feature"
        # TODO check if using RobustScaler is a better choice
        if dataset.null_count().sum_horizontal().sum() != 0:
            logger.warning(
                f"We have null values in the dataset: {dataset.null_count().sum_horizontal().sum()} as input"
            )
        if normalize_columns:
            dataset = dataset.with_columns(
                pl.col(self.input_features)
                .sub(pl.col(self.input_features).mean())
                .truediv(pl.col(self.input_features).std())
            )
        if len(monoton_smooth) > 0:
            dataset = dataset.sort(by=self.time_unit, descending=True)
            dataset = self.smooth_charge_cap_like(df=dataset, monoton_smooth=monoton_smooth)

        self.train_data: pl.DataFrame = dataset.filter(pl.col(self.time_unit) <= cutoff)
        self.target_data: pl.DataFrame = dataset.filter(pl.col(self.time_unit) > cutoff)
        # With multiple measurements per unit of time, we aggregate it to one value (the max) for the prediction
        # and the mean for the training
        self.target_data = self.target_data.group_by([
            self.col_id,
            self.time_unit,
        ]).agg(pl.col(target_features).max())
        self.train_data = self.train_data.group_by([self.col_id, self.time_unit]).agg(
            pl.col(input_features).mean()
        )
        # Ensure that every time_unit from 1 up to cutoff are filled
        device = self.train_data.select(self.col_id).unique()
        time_mapping = pl.DataFrame(
            {self.time_unit: np.arange(1, cutoff + 1)},
            schema={self.time_unit: self.train_data.schema[self.time_unit]},
        )
        device_time = device.join(time_mapping, how="cross")
        self.train_data = (
            device_time.join(other=self.train_data, on=[self.col_id, self.time_unit], how="left")
            .sort(by=[self.col_id, self.time_unit], descending=[True, False])
            .fill_null(strategy="forward")
        )
        if self.train_data.null_count().sum_horizontal().sum() != 0:
            logger.warning(
                f"Joining created null values: {self.train_data.null_count().sum_horizontal().sum()} as input.\nCheck that you have the correct {time_unit} recorded"
            )
            self.train_data = self.train_data.fill_null(strategy="backward")
            print(
                "You should assert that your input works for the use case (data per time slot)",
                flush=True,
            )
            raise ValueError(
                "Null values detected in train_data after filling all the slots. This indicates a data integrity issue.\nCheck that: first cycle is not null/NaN, you have a per device split"
            )

        ## Ensure that the devices have enough cycles to be trained on
        self.unique_ids = self.target_data.get_column(self.col_id).unique().to_list()
        self.train_data = self.train_data.filter(pl.col(self.col_id).is_in(self.unique_ids))

    def __len__(self):
        return self.target_data.shape[0]

    def __getitem__(self, index: int):
        """returns the training sample identified by index
        Args:
            index (int): Index into the different target's sample
        Returns:
            Dict[str:Tensor]: A HF-ready dict with input_ids and labels keys
        """
        y = None
        if self.stratify:
            y_device_name = self.unique_ids[index % len(self.unique_ids)]
            # Sample randomly a target data to predict from the device
            y = (
                self.target_data.filter(pl.col(self.col_id) == y_device_name)
                .sample(n=1)
                .row(0, named=True)
            )
        else:
            y = self.target_data.row(index, named=True)
        device_name = y[self.col_id]
        # Select the sequence corresponding to the device at the indexed target time
        X = self.train_data.filter(pl.col(self.col_id) == device_name)
        assert (
            X.shape[0] == self.cutoff
        ), f"You have a sequence of training that is not of size {self.cutoff}, it can be due to not having data for the first time_unit \n{X.shape = }"
        # Put the target time we are predicting into the input
        X = X.with_columns(pl.lit(y[self.time_unit]).alias(f"target_{self.time_unit}"))
        # Select all the columns we want and output a tuple coherent to the HF trainer expectations
        return {
            "input_ids": X.select(self.input_features + [f"target_{self.time_unit}"]).to_torch(
                dtype=pl.Float32
            ),
            "labels": pl.DataFrame(y)
            .select(self.target_features)
            .to_torch(dtype=pl.Float32)
            # view -1 because we have a 1xlen(target_features) tensor and when batching we want Bxlen(target_features)
            .view(-1),
        }

    def smooth_charge_cap_like(self, df: pl.DataFrame, monoton_smooth: List[str]) -> pl.DataFrame:
        """smooth the dataframe using the fact that the target is monotonically decreasing with outlier towards 0.
        By using a descending sorted dataset and using the cumulative max we can restore a nice curve.
        Args:
            df (pl.DataFrame): the df to operate on
            time_unit (str): the columns name containins the timestamp
            target_col (str): the column name to smooth
            ceiling (int, optional): the maximum value if you have some small upward outlier. Defaults to 1.1.

        Returns:
            pl.DataFrame: the smoothed data
        """
        dfs = []
        assert df.get_column(self.time_unit).is_sorted(descending=True)
        for device in df.get_column(self.col_id).unique():
            df_dev = df.filter(pl.col(self.col_id) == device).with_columns(
                pl.col(monoton_smooth).clip(0, 1.1).cum_max()
            )
            dfs.append(df_dev)
        df = pl.concat(dfs)
        return df


def hf2ttm(data: List[Dict], dev: torch.device) -> Dict:
    assert len(data) != 0
    first = data[0]
    k = list(first.keys())
    assert isinstance(first[k[0]], torch.Tensor) or isinstance(first[k[0]], np.ndarray)
    print(dev)
    batch = {}
    if "input_ids" in first.keys() and "labels" in first.keys():
        if isinstance(first["input_ids"], torch.Tensor):
            past_val = torch.stack([f["input_ids"] for f in data])
            batch["past_values"] = past_val.to(dev)
            print(past_val.device)
            print(past_val.to(dev).device)
            print(batch["past_values"].device)
            batch["future_values"] = torch.stack([f["labels"] for f in data]).to(dev)
        elif isinstance(first["input_ids"], np.ndarray):
            batch["past_values"] = torch.from_numpy(np.stack([f["input_ids"] for f in data])).to(
                dev
            )
            batch["future_values"] = torch.from_numpy(np.stack([f["labels"] for f in data])).to(dev)
        else:
            raise ValueError("Unable to stack non numpy / torch values")
        return batch
    di = default_data_collator(data)
    for k, v in di.items():
        di[k] = v.to(dev)
    return di


class HF2TMMCollator:
    def __init__(self, device):
        self.dev = device

    def __call__(self, features: List[Dict]) -> Dict:
        return hf2ttm(features, self.dev)
