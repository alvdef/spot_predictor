import torch.nn as nn


class SpotBiLSTM(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        self.hidden_size = model_config.get("hidden_size", 128)
        self.output_scale = model_config.get("output_scale", 1.0)
        self.input_size = model_config.get("input_size", 1)
        self.num_layers = model_config.get("num_layers", 2)  # Default: 2 layers
        self.dropout_rate = model_config.get("dropout_rate", 0.0)

        # Single LSTM with multiple layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                self.dropout_rate if self.num_layers > 1 else 0.0
            ),  # Apply dropout only if layers > 1
        )

        # Final dense layer
        self.dense = nn.Linear(self.hidden_size * 2, 1)

        # Initialize parameters
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        x, _ = self.lstm(x)  # First LSTM layer
        x = x[:, -1, :]  # Take only the last output
        x = self.dense(x)  # Dense layer
        x = x * self.output_scale  # Scale output

        return x
