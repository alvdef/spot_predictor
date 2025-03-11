from typing import Dict, Optional, List, Any
import torch
import json
import os
import logging


class Normalizer:
    """
    Pure PyTorch normalizer that handles per-instance normalization.
    Assumes all input data is already in tensor format.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize normalizer with optional device placement."""
        self.params: Dict[int, Dict[str, float]] = {}
        self.device = device
        self.logger = logging.getLogger(__name__)

    def fit(self, instance_id: int, values: torch.Tensor) -> "Normalizer":
        """
        Compute normalization parameters for an instance using torch operations.
        Expects data to already be a tensor.
        """
        if not isinstance(values, torch.Tensor):
            raise TypeError("Values must be a torch.Tensor")

        if self.device is not None:
            values = values.to(self.device)

        # Calculate parameters
        mean = float(values.mean().item())
        std = float(values.std().item())

        # Avoid division by zero
        if std < 1e-8:
            self.logger.warning(
                f"Near-zero standard deviation ({std}) for instance {instance_id}, "
                f"using 1.0 instead"
            )
            std = 1.0

        self.params[instance_id] = {"mean": mean, "std": std}
        return self

    def normalize(self, values: torch.Tensor, instance_ids: List[int]) -> torch.Tensor:
        """
        Normalize tensor values for given instance IDs, normalizing only the price feature (index 0).

        Args:
            values: Tensor to normalize
            instance_ids: List of instance IDs

        Returns:
            Normalized tensor with only the price feature normalized
        """
        if not isinstance(values, torch.Tensor):
            raise TypeError("Values must be a torch.Tensor")

        if not isinstance(instance_ids, list):
            raise TypeError("instance_ids must be a list of integers")

        if len(values) != len(instance_ids):
            raise ValueError("Number of values must match number of instance IDs")

        # Validate instance IDs
        for id in instance_ids:
            if id not in self.params:
                raise ValueError(f"No parameters for instance {id}")

        return self._apply_normalization(values, instance_ids, normalize=True)

    def denormalize(
        self, values: torch.Tensor, instance_ids: List[int]
    ) -> torch.Tensor:
        """
        Denormalize tensor values for given instance IDs.
        For 3D tensors, only denormalizes the price feature (index 0).
        For 1D/2D tensors, denormalizes all values (assumes they are prices).

        Args:
            values: Tensor to denormalize
            instance_ids: List of instance IDs

        Returns:
            Denormalized tensor
        """
        if not isinstance(values, torch.Tensor):
            raise TypeError("Values must be a torch.Tensor")

        if not isinstance(instance_ids, list):
            raise TypeError("instance_ids must be a list of integers")

        if len(values) != len(instance_ids):
            raise ValueError("Number of values must match number of instance IDs")

        for id in instance_ids:
            if id not in self.params:
                raise ValueError(f"No parameters for instance {id}")

        return self._apply_normalization(values, instance_ids, normalize=False)

    def _apply_normalization(
        self, values: torch.Tensor, instance_ids: List[int], normalize: bool
    ) -> torch.Tensor:
        """
        Apply normalization or denormalization to values.

        Args:
            values: Tensor to normalize/denormalize
            instance_ids: List of instance IDs
            normalize: If True, normalize; if False, denormalize

        Returns:
            Processed tensor
        """
        # Get normalization parameters for all instances in one go
        means = torch.tensor(
            [self.params[id]["mean"] for id in instance_ids], device=values.device
        )
        stds = torch.tensor(
            [self.params[id]["std"] for id in instance_ids], device=values.device
        )

        # Store original shape for reshaping later
        original_shape = values.shape
        result = values.clone()

        # Reshape parameters to match input dimensions for proper broadcasting
        if len(original_shape) == 3:  # [batch_size, seq_len, features]
            # Reshape means and stds to [batch_size, 1, 1] for proper broadcasting
            means = means.view(-1, 1, 1)
            stds = stds.view(-1, 1, 1)
        elif len(original_shape) == 2:  # [batch_size, seq_len]
            # Reshape means and stds to [batch_size, 1] for proper broadcasting
            means = means.view(-1, 1)
            stds = stds.view(-1, 1)

        # Apply normalization/denormalization
        if normalize:
            result = (values - means) / stds
        else:
            result = values * stds + means

        # Ensure output has the same shape as input
        if result.shape != original_shape:
            result = result.view(original_shape)

        return result

    def save(self, filepath: str) -> None:
        """
        Save normalization parameters to a JSON file.

        Args:
            filepath: Path to save the parameters

        Raises:
            IOError: If the file cannot be written
        """
        try:
            # Convert parameter dictionary keys to strings for JSON serialization
            serializable_params = {str(k): v for k, v in self.params.items()}

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(serializable_params, f, indent=2)

            self.logger.info(
                f"Saved normalization parameters for {len(self.params)} instances to {filepath}"
            )
        except IOError as e:
            self.logger.error(f"Failed to save normalization parameters: {str(e)}")
            raise

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> "Normalizer":
        """
        Load normalization parameters from a JSON file.

        Args:
            filepath: Path to the saved parameters
            device: Optional device to place tensors on

        Returns:
            A Normalizer instance with loaded parameters

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        normalizer = cls(device=device)

        try:
            with open(filepath, "r") as f:
                serialized_params = json.load(f)

            # Convert keys back to integers
            normalizer.params = {int(k): v for k, v in serialized_params.items()}

            normalizer.logger.info(
                f"Loaded normalization parameters for {len(normalizer.params)} instances from {filepath}"
            )
            return normalizer
        except FileNotFoundError:
            normalizer.logger.error(f"Normalizer file not found: {filepath}")
            raise
        except json.JSONDecodeError:
            normalizer.logger.error(f"Invalid JSON in normalizer file: {filepath}")
            raise

    def to(self, device: torch.device) -> "Normalizer":
        """
        Set the device for tensor operations.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')

        Returns:
            Self for method chaining
        """
        self.device = device
        return self

    def get_params_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the normalization parameters.

        Returns:
            Dictionary with summary statistics
        """
        if not self.params:
            return {"count": 0}

        means = [p["mean"] for p in self.params.values()]
        stds = [p["std"] for p in self.params.values()]

        return {
            "count": len(self.params),
            "mean_stats": {
                "min": min(means),
                "max": max(means),
                "avg": sum(means) / len(means),
            },
            "std_stats": {
                "min": min(stds),
                "max": max(stds),
                "avg": sum(stds) / len(stds),
            },
        }

    def get_params(self) -> Dict[str, Any]:
        """
        Get all normalization parameters.

        Returns:
            Dictionary with all instance parameters
        """
        return {str(k): v for k, v in self.params.items()}

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set normalization parameters.

        Args:
            params: Dictionary with parameters
        """
        self.params = {int(k): v for k, v in params.items()}
