from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn

from .base import Model
from utils import load_config


class SpotGRU(Model):
    REQUIRED_FIELDS = [
        "hidden_size",
        "output_scale",
        "input_size",
        "num_layers",
        "prediction_length",
    ]

    def __init__(
        self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None
    ):
        super().__init__()

        if config_path:
            self.config = load_config(config_path, "model_config", self.REQUIRED_FIELDS)
        elif config_dict:
            self.config = config_dict
            # Validate required fields
            missing_fields = [
                field for field in self.REQUIRED_FIELDS if field not in config_dict
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in config_dict: {missing_fields}"
                )
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self._build_model(self.config)
        self.initialized = True

    def _build_model(self, config: Dict[str, Any]) -> None:
        # GRU encoder
        self.gru = nn.GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=True,
        )

        # Decoder for multi-step output
        self.decoder = nn.Linear(config["hidden_size"], config["prediction_length"])

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self) -> None:
        # Initialize GRU weights using orthogonal initialization for better gradient flow
        # This helps with training stability in recurrent networks
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        # For the decoder, use Kaiming initialization which works well for linear layers
        # followed by non-linearities (if you add any later)
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity="linear")
        nn.init.constant_(self.decoder.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input sequence of shape (batch_size, sequence_length)
               or (batch_size, sequence_length, features)

        Returns:
            Predictions of shape (batch_size, prediction_length)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Add feature dimension

        # Process through GRU encoder
        _, hidden = self.gru(x)

        # Get last hidden state from the top layer
        last_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Decode to get multi-step prediction
        output = self.decoder(last_hidden)  # (batch_size, prediction_length)

        output = output * self.config["output_scale"]

        return output
