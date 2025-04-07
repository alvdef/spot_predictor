from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn

from .base import Model


class LSTM(Model):
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
        # LSTM encoder
        self.lstm = nn.LSTM(
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
            self.combiner = nn.Linear(config["hidden_size"] * 2, config["hidden_size"])

        # Decoder for multi-step output
        self.decoder = nn.Linear(config["hidden_size"], config["prediction_length"])

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

        if self.feature_size > 0:
            for m in self.feature_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.kaiming_normal_(self.combiner.weight, nonlinearity="relu")
            nn.init.zeros_(self.combiner.bias)

    def forward(
        self, x: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor or tuple of (sequence, features)
            target: Optional target tensor (not used in this model, but included for API consistency)

        Returns:
            Predictions tensor
        """
        # Extract sequence and features
        if isinstance(x, tuple):
            sequence, features = x
        else:
            sequence, features = x, None

        if len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(-1)

        # Get the last LSTM hidden state
        _, (hidden, _) = self.lstm(sequence)
        last_hidden = hidden[-1]

        # Process features if available
        if features is not None and self.feature_size > 0:
            processed_features = self.feature_net(features)
            final_hidden = self.combiner(
                torch.cat([last_hidden, processed_features], dim=1)
            )
        else:
            final_hidden = last_hidden

        # Generate prediction
        return self.decoder(final_hidden) * self.config["output_scale"]
