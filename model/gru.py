from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn

from .base import Model


class GRU(Model):
    REQUIRED_FIELDS = [
        "hidden_size",
        "output_scale",
        "num_layers",
        "tr_prediction_length",
        "feature_size",
    ]

    def __init__(self, work_dir: str):
        super().__init__(work_dir)

    def _build_model(self, config: Dict[str, Any]) -> None:
        # GRU encoder
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=True,
        )

        # Decoder for multi-step output
        self.decoder = nn.Linear(config["hidden_size"], config["tr_prediction_length"])

        self._initialize_weights()
        self.to(self.device)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input sequence of shape (batch_size, sequence_length)
               or (batch_size, sequence_length, features)
            target: Optional target tensor (not used in this model, but included for API consistency)

        Returns:
            Predictions of shape (batch_size, prediction_length)
        """
        sequence = x[0]

        if len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(-1)  # Add feature dimension

        # Process through GRU encoder
        _, hidden = self.gru(sequence)
        final_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Decode to get multi-step prediction
        output = self.decoder(final_hidden)
        output = output * self.config["output_scale"]

        return output
