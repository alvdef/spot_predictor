import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils import get_device, load_config


class Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.device = get_device()

    @abstractmethod
    def forward(self, x):
        pass

    @torch.no_grad()
    def forecast(self, sequence, n_steps):
        """
        Generate recursive predictions for n_steps ahead.

        Args:
            sequence (torch.Tensor): Input sequence of shape (batch_size, sequence_length)
                                   or (batch_size, sequence_length, 1)
            n_steps (int): Number of steps to predict

        Returns:
            torch.Tensor: Predictions of shape (batch_size, n_steps)
        """
        self.eval()

        if len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(-1)

        batch_size = sequence.shape[0]
        predictions = torch.zeros((batch_size, n_steps), device=self.device)
        current_sequence = sequence

        for step in range(n_steps):
            output = self(current_sequence)
            predictions[:, step] = output.squeeze()
            current_sequence = torch.cat(
                [current_sequence[:, 1:, :], output.unsqueeze(1)], dim=1
            )

        return predictions


class SpotLSTM(Model):
    REQUIRED_FIELDS = [
        "hidden_size",
        "output_scale",
        "input_size",
        "num_layers",
        "dropout_rate",
    ]

    def __init__(self, config_path: str):
        super().__init__()

        self.config = load_config(config_path, "model_config", self.REQUIRED_FIELDS)

        self.lstm = nn.LSTM(
            hidden_size=self.config["hidden_size"],
            input_size=self.config["input_size"],
            num_layers=self.config["num_layers"],
            batch_first=True,
            dropout=(
                self.config["dropout_rate"] if self.config["num_layers"] > 1 else 0.0
            ),
        )

        self.dense = nn.Linear(self.config["hidden_size"], 1)
        self._initialize_weights()
        # Move model to the correct device
        self.to(self.device)

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        x = x * self.config["output_scale"]

        return x
