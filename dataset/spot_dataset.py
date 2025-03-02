from typing import Union, Dict, Tuple, Optional, List, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging

from utils import get_device, load_config


class SpotDataset(Dataset):
    """Dataset for spot price prediction using sliding windows of time series data."""

    REQUIRED_FIELDS = [
        "sequence_length",
        "prediction_length",
        "window_step",
        "batch_size",
    ]

    def __init__(self, df: pd.DataFrame, config_path: str = "config.yaml"):
        """Initialize dataset with all data directly on GPU."""
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")

        if "spot_price" not in df.columns or "id_instance" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'spot_price' and 'id_instance' columns"
            )

        self.config = load_config(config_path, "dataset_config", self.REQUIRED_FIELDS)
        self.device = get_device()

        self.X, self.y, self.instance_ids = self._create_sequences(df)

        logging.info(f"Dataset created with {len(self.X)} sequences on {self.device}")
        logging.info(f"Input shape: {self.X.shape}, Target shape: {self.y.shape}")

    def _create_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sequence_length = self.config["sequence_length"]
        prediction_length = self.config["prediction_length"]
        window_step = self.config["window_step"]
        required_length = sequence_length + prediction_length

        # Pre-calculate sequence counts to allocate memory efficiently
        total_sequences = 0
        instance_counts = {}

        for instance_id, group in df.groupby("id_instance"):
            n_values = len(group)
            if n_values < required_length:
                continue

            n_sequences = (n_values - required_length) // window_step + 1
            instance_counts[instance_id] = n_sequences
            total_sequences += n_sequences

        if total_sequences == 0:
            raise ValueError("No valid sequences could be created")

        # Pre-allocate tensors on CPU initially for better memory efficiency
        X = np.zeros((total_sequences, sequence_length), dtype=np.float32)
        y = np.zeros((total_sequences, prediction_length), dtype=np.float32)
        ids = []

        # Create sequences efficiently with a single pass per instance
        idx = 0

        for instance_id, group in df.groupby("id_instance"):
            if instance_id not in instance_counts:
                continue

            values = group["spot_price"].values.astype(np.float32)

            # Use NumPy's efficient array operations
            for seq_idx in range(instance_counts[instance_id]):
                start_idx = seq_idx * window_step
                end_x_idx = start_idx + sequence_length
                end_y_idx = end_x_idx + prediction_length

                if end_y_idx > len(values):
                    break

                X[idx] = values[start_idx:end_x_idx]
                y[idx] = values[end_x_idx:end_y_idx]
                ids.append(instance_id)
                idx += 1

        # Adjust tensors if we didn't use all pre-allocated space
        if idx < total_sequences:
            X = X[:idx]
            y = y[:idx]

        # Move to GPU in one batch transfer rather than many small transfers
        X = (
            torch.from_numpy(X).to(self.device).unsqueeze(-1)
        )  # Add feature dimension at the end
        y = torch.from_numpy(y).to(self.device)

        logging.info(
            f"Created {len(ids)} sequences from {len(instance_counts)} instances"
        )

        return X, y, ids

    def get_sequences(self, instance_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get sequence data filtered by instance ID.

        Args:
            instance_id: If provided, only returns data for this instance.
                         Otherwise, returns all sequences.

        Returns:
            Dictionary containing sequences, targets, and optionally instance IDs

        Raises:
            ValueError: If the requested instance ID doesn't exist in the dataset
        """
        if instance_id is not None:
            # Use tensor indexing for better performance
            mask = torch.tensor(
                [iid == instance_id for iid in self.instance_ids], device=self.device
            )
            if not mask.any():
                raise ValueError(f"Instance ID {instance_id} not found in dataset")

            return {"sequences": self.X[mask], "targets": self.y[mask]}

        return {
            "sequences": self.X,
            "targets": self.y,
            "instance_ids": self.instance_ids,
        }

    def get_data_loader(self, shuffle: bool = True) -> DataLoader:
        """Get DataLoader for GPU data."""
        return DataLoader(
            self,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            pin_memory=True,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def get_instance_id(self, idx: int) -> int:
        return self.instance_ids[idx]
