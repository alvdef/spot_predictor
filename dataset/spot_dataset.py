from typing import Optional, Union, Dict, Tuple, List, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from utils import get_device, load_config, get_logger, extract_time_features


class SpotDataset(Dataset):
    """Dataset for spot price prediction using sliding windows of time series data.

    This dataset generates sequences of historical spot prices combined with two types of features:
    1. Instance features: Characteristics of each AWS instance type (memory, size, etc.)
    2. Time features: Temporal features extracted from timestamps (day of week, month, etc.)

    The DataLoader returns items in the following format:
        (input_data, target_tensor)

    Where input_data is a tuple with three components:
        (sequence_tensor, instance_features_tensor, time_features_tensor)

    And:
        - sequence_tensor: Shape (sequence_length, 1) containing price history
        - instance_features_tensor: Shape (1, feature_size) containing instance features
        - time_features_tensor: Shape (sequence_length, time_feature_size) with time features
        - target_tensor: Shape (prediction_length) containing target prices

    Models can choose which features to use by selecting elements from the input tuple.
    For example:
        - Basic model: Use only sequences[0]
        - Instance-aware model: Use sequences[0] and sequences[1]
        - Time-aware model: Use sequences[0] and sequences[2]
        - Full-featured model: Use all three elements
    """

    REQUIRED_FIELDS = [
        "sequence_length",
        "prediction_length",
        "tr_prediction_length",
        "window_step",
        "batch_size",
        "instance_features",
        "time_features",
    ]
    TIME_COL = "price_timestamp"

    def __init__(
        self,
        df: pd.DataFrame,
        instance_features_df: pd.DataFrame,
        work_dir: str,
        training: bool = False,
    ):
        """
        Initialize dataset with all data directly on GPU.

        Args:
            df: DataFrame containing spot price data
            work_dir: Working directory containing config file
            instance_features_df: DataFrame containing instance features
            mode: s
        """
        self.logger = get_logger(__name__)
        if df.empty:
            raise ValueError("DataFrame cannot be None or empty")

        if "spot_price" not in df.columns or "id_instance" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'spot_price' and 'id_instance' columns"
            )

        if self.TIME_COL not in df.columns:
            raise ValueError(
                f"Timestamp column '{self.TIME_COL}' required but not found in DataFrame"
            )

        self.config = load_config(
            f"{work_dir}/config.yaml", "dataset_config", self.REQUIRED_FIELDS
        )
        self.device = get_device()
        self.training = training
        self.instance_features_df = instance_features_df

        self.features_tensor, self.feature_mapping = self._process_instance_features()
        (
            self.X,
            self.y,
            self.instance_ids,
            self.timestamps,
            self.time_features,
        ) = self._create_sequences(df)

        self.logger.info(f"Dataset created with {len(self.X)} sequences")
        self.logger.info(f"Context shape: {self.X.shape}, Target shape: {self.y.shape}")
        self.logger.info(f"Instance feature shape: {self.features_tensor.shape}")
        self.logger.info(f"Time feature shape: {self.time_features.shape}")

    def _process_instance_features(self) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Process instance features based on config specifications.

        Only processes features listed in config["instance_features"].
        Handles both standard categorical features and list-type features.

        Returns:
            Tuple containing:
                - Features tensor (one-hot encoded)
                - Mapping from instance ID to tensor index
        """
        feature_cols = self.config["instance_features"]
        filtered_df = self.instance_features_df[feature_cols].copy()

        # Initialize the features dataframe with instance IDs as index
        features_df = pd.DataFrame(index=filtered_df.index)

        # Process each feature column according to its type
        for col in feature_cols:
            col_values = filtered_df[col]

            # Check if column contains list values
            if any(isinstance(x, list) for x in col_values if x is not None):
                # Extract unique values across all lists
                unique_values = set()
                for value_list in col_values:
                    if isinstance(value_list, list):
                        unique_values.update(value_list)

                # Create binary features for each unique value
                for value in unique_values:
                    features_df[f"{col}_{value}"] = col_values.apply(
                        lambda x: True if isinstance(x, list) and value in x else False
                    )
            else:
                # Handle standard categorical features with one-hot encoding
                encoded = pd.get_dummies(col_values, prefix=col, drop_first=False)
                features_df = pd.concat([features_df, encoded], axis=1)

        # Create mapping from instance ID to feature index
        instance_ids = self.instance_features_df.index.tolist()
        feature_mapping = {
            instance_id: idx for idx, instance_id in enumerate(instance_ids)
        }
        features_tensor = torch.tensor(features_df.values, dtype=torch.float32)

        return features_tensor, feature_mapping

    def _create_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[
        torch.Tensor, torch.Tensor, List[int], List[List[pd.Timestamp]], torch.Tensor
    ]:
        """
        Create sequences for training and evaluation with timestamps and time features.

        Args:
            df: DataFrame with price, instance ID, and timestamp columns

        Returns:
            Tuple containing:
                - X: Input sequence tensor
                - y: Target sequence tensor
                - instance_ids: List of instance IDs for each sequence
                - timestamps: List of timestamp lists for each sequence
                - time_features: Tensor of time features
        """
        sequence_length = self.config["sequence_length"]
        prediction_length = (
            self.config["tr_prediction_length"]
            if self.training
            else self.config["prediction_length"]
        )
        window_step = self.config["window_step"]
        required_length = sequence_length + prediction_length

        # Use list comprehensions for better efficiency
        sequences = []
        targets = []
        ids = []
        timestamps_list = []
        time_features_list = []

        # Group by instance_id for easier processing
        grouped_df = df.groupby("id_instance")

        group_lengths = grouped_df.size()
        min_len = group_lengths.min()
        max_len = group_lengths.max()

        self.logger.info(
            f"Minimum required length for sequences {required_length}. Min group length: {min_len}, Max group length: {max_len}"
        )

        for instance_id, group in grouped_df:
            # Skip instances with insufficient data points
            if len(group) < required_length:
                self.logger.warning(
                    f"Skipped instance {instance_id} on building sequences"
                )
                continue

            # Sort and extract values and timestamps
            group = group.sort_values(by=self.TIME_COL)
            values = group["spot_price"].values.astype(np.float32)
            time_values = group[self.TIME_COL]

            # Vectorized sliding windows for values
            windows = sliding_window_view(values, required_length)[::window_step]
            X_group = windows[:, :sequence_length]
            Y_group = windows[:, sequence_length:]

            # Precompute time features for all timestamps once
            full_time_feat = np.array(
                extract_time_features(time_values, self.config["time_features"]),
                dtype=np.float32,
            )
            time_feat_windows = sliding_window_view(
                full_time_feat, sequence_length, axis=0
            )[::window_step]

            # Collect sequences
            for i in range(X_group.shape[0]):
                sequences.append(X_group[i])
                targets.append(Y_group[i])
                ids.append(instance_id)
                # Full timestamps for input+target
                seq_timestamps = time_values.iloc[
                    i * window_step : i * window_step + required_length
                ].tolist()
                timestamps_list.append(seq_timestamps)
                # Corresponding time features
                time_features_list.append(time_feat_windows[i])

        if not sequences:
            raise ValueError("No valid sequences could be created")

        X_tensor = torch.tensor(
            np.array(sequences, dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(-1)
        y_tensor = torch.tensor(
            np.array(targets, dtype=np.float32), dtype=torch.float32
        )
        time_features_tensor = torch.tensor(
            np.array(time_features_list, dtype=np.float32),
            dtype=torch.float32,
        )

        self.logger.info(f"Created {len(ids)} sequences from {len(set(ids))} instances")

        return X_tensor, y_tensor, ids, timestamps_list, time_features_tensor

    def get_sequences(self, instance_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get sequence data filtered by instance ID.

        Args:
            instance_id: If provided, only returns data for this instance.
                         Otherwise, returns all sequences.

        Returns:
            Dictionary containing:
            - sequences: Price history tensors
            - targets: Target price tensors
            - instance_ids: List of instance IDs (if no instance_id provided)
            - timestamps: Timestamp lists for debugging
            - time_features: Time feature tensors
            - instance_features: Instance feature tensor (if instance_id provided)

        Raises:
            ValueError: If the requested instance_id doesn't exist in the dataset
        """
        if instance_id is not None:
            # Use tensor indexing for better performance
            mask = torch.tensor([iid == instance_id for iid in self.instance_ids])
            if not mask.any():
                raise ValueError(f"Instance ID {instance_id} not found in dataset")

            # Get feature index for this instance
            feature_idx = self.feature_mapping[instance_id]
            instance_features = self.features_tensor[feature_idx]

            # Convert mask to CPU numpy for timestamp indexing
            cpu_mask = mask.cpu().numpy()

            return {
                "sequences": self.X[mask],
                "targets": self.y[mask],
                "timestamps": [
                    self.timestamps[i] for i in range(len(cpu_mask)) if cpu_mask[i]
                ],
                "time_features": self.time_features[mask],
                "instance_features": instance_features,
            }

        return {
            "sequences": self.X,
            "targets": self.y,
            "instance_ids": self.instance_ids,
            "timestamps": self.timestamps,
            "time_features": self.time_features,
        }

    def get_data_loader(self, shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader that yields batches of (input_tuple, target) pairs.

        The input_tuple contains (sequence, instance_features, time_features).
        Models can choose which elements of this tuple to use.

        Args:
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader configured with appropriate batch size
        """
        return DataLoader(
            self,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            pin_memory=True,
            num_workers=0,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Get item at the specified index with all features.

        Args:
            idx: Index of the item to retrieve

        Returns:
            A tuple containing:
                - Input tuple (sequence, instance_features, time_features)
                - Target tensor
        """
        instance_id = self.instance_ids[idx]
        sequence = self.X[idx]
        target = self.y[idx]

        # Get instance features
        feature_idx = self.feature_mapping[instance_id]
        instance_feats = self.features_tensor[feature_idx].unsqueeze(0)

        # Get time features
        time_feats = self.time_features[idx]

        # Return all features in a consistent format
        # Models can choose which elements to use
        return (sequence, instance_feats, time_feats), target

    def get_instance_id(self, idx: int) -> int:
        """Get the instance ID for a sequence at the given index."""
        return self.instance_ids[idx]

    def get_timestamps(self, idx: int) -> List[pd.Timestamp]:
        """Get the timestamps associated with a sequence."""
        return self.timestamps[idx]

    @property
    def len_time_features(self) -> int:
        """Get the innermost dimension of time features."""
        return self.time_features.shape[1]

    @property
    def time_features_shape(self) -> tuple:
        """Get the full shape of time features (excluding batch dimension)."""
        return (self.time_features.shape[1], self.time_features.shape[2])

    @property
    def len_instance_features(self) -> int:
        """Get the time features associated with a sequence."""
        return len(self.features_tensor[0])

    def get_instance_indices(self, instance_id: int) -> List[int]:
        """
        Get all indices corresponding to a specific instance.

        Args:
            instance_id: The instance ID to filter for

        Returns:
            List of indices in the dataset that correspond to the specified instance

        Raises:
            ValueError: If the instance ID is not found in the dataset
        """
        indices = [i for i, iid in enumerate(self.instance_ids) if iid == instance_id]
        if not indices:
            raise ValueError(f"Instance ID {instance_id} not found in dataset")
        return indices

    def get_instance_dataset(self, instance_id: int) -> Subset:
        """
        Create a subset dataset containing only data for the specified instance.

        This uses PyTorch's Subset class to provide an efficient view of the data
        without duplicating storage.

        Args:
            instance_id: The instance ID to filter for

        Returns:
            A Subset of this dataset containing only the specified instance

        Raises:
            ValueError: If the instance ID is not found in the dataset
        """
        indices = self.get_instance_indices(instance_id)
        return Subset(self, indices)

    def get_instance_dataloader(
        self, instance_id: int, batch_size: Optional[int] = None, shuffle: bool = False
    ) -> DataLoader:
        """
        Get a DataLoader that yields batches only for the specified instance.

        Args:
            instance_id: The instance ID to filter for
            batch_size: Batch size for the DataLoader (defaults to the config batch size)
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader configured with appropriate batch size for the specified instance

        Raises:
            ValueError: If the instance ID is not found in the dataset
        """
        instance_dataset = self.get_instance_dataset(instance_id)
        return DataLoader(
            instance_dataset,
            batch_size=batch_size or self.config["batch_size"],
            shuffle=shuffle,
            pin_memory=True,
        )

    def get_unique_instance_ids(self) -> List[int]:
        """
        Get a list of all unique instance IDs in the dataset.

        Returns:
            List of unique instance IDs
        """
        return list(set(self.instance_ids))
