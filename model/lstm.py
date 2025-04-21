from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn

from .base import Model


class LSTM(Model):
    REQUIRED_FIELDS = [
        "hidden_size",
        "output_scale",
        "input_size",
        "num_layers",
        "tr_prediction_length",
    ]

    def __init__(self, work_dir: str):
        super().__init__(work_dir)

    def _build_model(self, config: Dict[str, Any]) -> None:
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=True,
        )

        # Decoder for multi-step output
        self.decoder = nn.Linear(config["hidden_size"], config["tr_prediction_length"])

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self) -> None:
        # Initialize LSTM weights for better training stability
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                # Set forget gate bias to 1.0 to reduce vanishing gradients
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        # Initialize decoder and feature network
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.decoder.bias)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor or tuple of (sequence, features)
            target: Optional target tensor (not used in this model, but included for API consistency)

        Returns:
            Predictions tensor
        """
        sequence = x[0]

        if len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(-1)

        # Get the last LSTM hidden state
        _, (hidden, _) = self.lstm(sequence)
        final_hidden = hidden[-1]

        # Generate prediction
        return self.decoder(final_hidden) * self.config["output_scale"]
