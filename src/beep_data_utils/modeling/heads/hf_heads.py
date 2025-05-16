"""This module defines models following the hugging face API, all of them are ready to be used with a HF-trainer

They all MUST return a tuple or an subclass of modelOutput,
The input variable name in the forward methods follows the output of the dataset (see hf_utils datasets)
The model can compute the loss if a `labels` argument is provided and that loss is returned as the first element of the tuple (if your model returns tuples)

Each model is defined with its config"""

from typing import Any, Dict, List, Optional

from torch import Tensor, nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel


class LSTMRegressorConfig(PretrainedConfig):
    """This class is the configuration for the LSTMRegressor model,
    It controls mostly 4 things:
        - The size of the input embedding (num_input) (if you produce embeddings with a different Fondation Model's config, please adapt it)
        - The size of the hidden representation in the LSTM layers (hidden_size)
        - The number of regressed channel (num_output_feature)
        - The number of layers we use
    """

    def __init__(
        self,
        num_input_feature: int = 256,
        hidden_size: int = 32,
        num_output_feature: int = 3,
        model_type: str = "LSTM",
        num_layers: int = 2,
        **kwargs,
    ):
        """Initialize the config

        Args:
            num_input_feature (int, optional): The number of input feature this . Defaults to 256.
            hidden_size (int, optional): The size of the hidden embedding in the LSTM layer . Defaults to 32.
            num_output_feature (int, optional): The number of output channel to predict. Defaults to 3.
            model_type (str, optional): The model type. Defaults to "LSTM".
            num_layers (int, optional): The number of layer in the LSTM stack
        """
        super().__init__(**kwargs)
        self.num_input_feature = num_input_feature
        self.hidden_size = hidden_size
        self.num_output_feature = num_output_feature
        self.model_type = model_type
        self.num_layers = num_layers


class LSTMRegressor(PreTrainedModel):
    """LSTM Regressor head.

    Each device is represented by a list of size n embeddings representing the first n elements in the sequence,
    You should have concatenated to these the target profile you want to predict. A target profile is information about the timestep out of which we are trying to predict some values
    """

    config = LSTMRegressorConfig

    def __init__(self, config: LSTMRegressorConfig):
        """Initialize LSTMRegressor.
        Args:
           config: the configuration for the LSTMRegressor, see its own doc for more information
        """
        super().__init__(config=config)
        self.input_dim = config.num_input_feature
        self.num_layers = config.num_layers
        # Number of channel to predict from the embeddings
        self.hidden_size = config.hidden_size
        self.output_dim = config.num_output_feature
        self.stacked_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.regressor = nn.Linear(in_features=self.hidden_size, out_features=self.output_dim)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        return_loss=True,
    ) -> Dict[str, Tensor]:
        """
        Compute a forward pass trought the network
        The network is composed of the LSTM stack followed by a SiLU activation and a linear projection to the regressed values
        The loss is computed if the labels argument is not None.
        Args:
            input_ids (Tensor): is a tensor of shape batch cutoff embeding_dim where the cutoff is how many element in the sequence we handle
            labels (Optional[Tensor], optional): The correct value, if not None the loss is computed. Defaults to None.
            return_loss (bool, optional): To be able to use with the Trainer API, not currently used. Defaults to True.
        Returns:
            Dict[str, Tensor]: A dictionary with a logits key and an optional loss key. The loss key is present if labels is not None
        """
        assert input_ids.dim() == 3, f"Got a batch with too many dimension: {input_ids.shape=}"
        out, (_, _) = self.stacked_lstm(input_ids)
        # Take the last output which encapsulate the whole sequence,
        fused_seq = out[:, -1, :]
        # Activation + Linear
        logit = self.regressor(F.silu(fused_seq.view(input_ids.shape[0], self.hidden_size)))
        if labels is None:
            return {"logits": logit}
        loss = nn.functional.mse_loss(logit, labels)
        return {"loss": loss, "logits": logit}


class LinearRegressorConfig(PretrainedConfig):
    """This model is the configuration for the LinearRegressor model,
    It controls mostly 4 things:
        - The size of the input embedding (num_input_feature) (if you produce embeddings with a different Fondation Model's config)
        - The sequence length (seq_len)
        - The number of regressed channel (num_output_feature)
        - The number of layers
    Notice that each layer halfs the input by half, if you have an input of size 1024, you can use at most 10 layers (1024 // (2^num_layer) must be greater than 0 where // is the python integer division)
    """

    def __init__(
        self,
        num_input_feature: int = 256,
        seq_len: int = 100,
        num_output_feature: int = 3,
        num_layers: int = 1,
        model_type: str = "Linear",
        **kwargs,
    ):
        """Initialize the configuration

        Args:
            num_input_feature (int, optional): The number of input feature, should be embedding dim * sequence length. Defaults to 256.
            seq_len (int, optional): The length of the sequence . Defaults to 100.
            num_output_feature (int, optional): Number of output feature. Defaults to 3.
            model_type (str, optional): The model type. Defaults to "Linear".
            num_layers (int,optional):  The number of layer, each layer input size is previous layer outputsize // 2
        """
        super().__init__(**kwargs)
        self.model_type = model_type
        self.num_input_feature = num_input_feature
        self.seq_len = seq_len
        self.num_output_feature = num_output_feature
        self.num_layers = num_layers


class LinearRegressor(PreTrainedModel):
    """Linear Regressor head.

    Each device is represented by a list of size n embeddings representing the first n recordings/measurment from the device,
    You should have concatenated to these the target profile you want to predict. A target profile is information about the timestep on which we are trying to predict some values.
    """

    config = LinearRegressorConfig

    def __init__(self, config: LinearRegressorConfig):
        """Initialize the Linear Regressor.
        Args:
           config: the configuration for the LinearRegressor, see its own doc for more information
        """
        super().__init__(config=config)
        self.input_dim = config.num_input_feature
        # Number of channel to predict from the embeddings
        self.seq_len = config.seq_len
        self.output_dim = config.num_output_feature
        if config.num_layers == 1:
            self.proj = nn.Linear(
                in_features=self.input_dim * self.seq_len, out_features=self.output_dim, bias=False
            )
        else:
            layers: List[Any] = []
            current_dim = self.input_dim * self.seq_len
            for _ in range(config.num_layers - 1):
                next_dim = current_dim // 2
                layers.append(nn.Linear(current_dim, next_dim))
                layers.append(nn.GELU())
                current_dim = next_dim
            # Add the final layer with no bias
            layers.append(nn.Linear(current_dim, self.output_dim, bias=False))

            self.proj = nn.Sequential(*layers)  # type: ignore[assignment]

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        return_loss=True,
    ) -> Dict[str, Tensor]:
        """Compute a forward pass trought the network
        This is a linear projection to the regressed values
        Args:
            input_ids (Tensor): Tensor of shape batch seq_len embeding_dim where the seq_len is how many element in the sequence we handle
            labels (Optional[Tensor], optional): The correct values (also known as ground truth), if not None the loss is computed. Defaults to None.
            return_loss (bool, optional): To be able to use with the Trainer API, not currently used. Defaults to True.
        Returns:
            Dict[str, Tensor]: A dictionary with a logits key and an optional loss key. The logits have a shape of batch regressed_channel The loss key is present if labels is not None. If
        """
        assert input_ids.dim() == 3, f"Got a batch with too many dimension: {input_ids.shape=}"
        logit = self.proj(input_ids.view(-1, self.input_dim * self.seq_len))
        if labels is None:
            return {"logits": logit}
        loss = nn.functional.mse_loss(logit, labels)
        return {"loss": loss, "logits": logit}
