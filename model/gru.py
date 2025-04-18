from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn

from .base import Model


class GRU(Model):
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
