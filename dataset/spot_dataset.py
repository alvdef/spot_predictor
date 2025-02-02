import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import get_device, load_config


class SpotDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config_path: str = "config.yaml"):
        required_fields = ["sequence_length", "window_step", "batch_size"]
        self.config = load_config(config_path, "dataset_config", required_fields)

        X, y = self._create_sequences(df)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        device = get_device()
        if device:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def get_data_loader(self, shuffle=True, num_workers=0, pin_memory=False):
        """Get data loader with configurable parameters"""
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _create_sequences(self, df: pd.DataFrame):
        sequence_length = self.config["sequence_length"]
        window_step = self.config["window_step"]
        prediction_length = 1
        all_X, all_y = [], []

        min_required_length = sequence_length + prediction_length

        for id_instance in df["id_instance"].unique():
            prices = df[df["id_instance"] == id_instance]["spot_price"].values

            if len(prices) < min_required_length:
                continue

            max_start_idx = len(prices) - min_required_length
            for i in range(0, max_start_idx + 1, window_step):
                seq_x = prices[i : i + sequence_length]
                seq_y = prices[
                    i + sequence_length : i + sequence_length + prediction_length
                ]

                if len(seq_x) == sequence_length and len(seq_y) == prediction_length:
                    all_X.append(seq_x)
                    all_y.append(seq_y)

        X = np.array(all_X)
        y = np.array(all_y)

        assert len(X) == len(y), "Mismatched X and y lengths"
        assert (
            X.shape[1] == sequence_length
        ), f"Wrong input sequence length: {X.shape[1]}"
        assert (
            y.shape[1] == prediction_length
        ), f"Wrong target sequence length: {y.shape[1]}"

        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(-1), self.y[idx]
