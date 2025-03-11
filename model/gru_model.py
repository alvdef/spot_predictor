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
        "feature_size",
    ]

    def __init__(self, work_dir: str):
        super().__init__(work_dir)

    def _build_model(self, config: Dict[str, Any]) -> None:
        # GRU encoder
        self.gru = nn.GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=True,
        )

        # Feature processing network (if features are used)
        self.feature_size = config.get("feature_size", 0)
        if self.feature_size > 0:
            self.feature_net = nn.Sequential(
                nn.Linear(self.feature_size, config["hidden_size"]),
                nn.ReLU(),
                nn.Linear(config["hidden_size"], config["hidden_size"]),
                nn.ReLU(),
            )

            # Combining layer to merge GRU and feature outputs
            self.combiner = nn.Linear(config["hidden_size"] * 2, config["hidden_size"])

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

        # Initialize feature network if it exists
        if self.feature_size > 0:
            for m in self.feature_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

            nn.init.kaiming_normal_(self.combiner.weight, nonlinearity="relu")
            nn.init.constant_(self.combiner.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input sequence of shape (batch_size, sequence_length)
               or (batch_size, sequence_length, features)

        Returns:
            Predictions of shape (batch_size, prediction_length)
        """
        # Handle sequence and optional features
        if isinstance(x, tuple):
            sequence, features = x
        else:
            sequence, features = x, None

        if len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(-1)  # Add feature dimension

        # Process through GRU encoder
        _, hidden = self.gru(sequence)
        last_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Process features if available
        if features is not None and self.feature_size > 0:
            processed_features = self.feature_net(features)
            combined = torch.cat([last_hidden, processed_features], dim=1)
            final_hidden = self.combiner(combined)
        else:
            final_hidden = last_hidden

        # Decode to get multi-step prediction
        output = self.decoder(final_hidden)
        output = output * self.config["output_scale"]

        return output
