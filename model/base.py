from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils import get_device


class Model(nn.Module, ABC):
    """
    Model is an abstract base class for neural network models that extends PyTorch's nn.Module.
    It provides a common interface for models, including methods for forward pass and forecasting.

    Attributes:
        device (torch.device): The device (CPU or GPU) on which the model is running.
        normalizer: The normalizer object for handling data normalization and denormalization.
        config (Dict): Configuration parameters for the model.

    Methods:
        forward(x):
            Abstract method to be implemented by subclasses for the forward pass of the model.

        forecast(sequence, n_steps, instance_id=None):
            Generate recursive predictions for a specified number of steps ahead.

        predict(x):
            Makes a single forward pass through the model.

        save(path):
            Save model weights and normalizer parameters.

        load(path):
            Load model weights and normalizer parameters.
    """

    def __init__(self):
        super().__init__()
        self.device = get_device()
        self.normalizer = None
        self.config = {}

    def attach_normalizer(self, normalizer) -> None:
        """
        Attach a normalizer object to the model for data normalization/denormalization.

        Args:
            normalizer: Normalizer object that implements normalize and denormalize methods
        """
        self.normalizer = normalizer

    @abstractmethod
    def _build_model(self, config: Dict[str, Any]) -> None:
        """
        Build the model architecture based on configuration parameters.

        Args:
            config: Dictionary containing model architecture parameters
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def predict(
        self, x: torch.Tensor, instance_ids: Optional[Union[int, List[int]]] = None
    ) -> torch.Tensor:
        """
        Make predictions with a single forward pass (no recursion).

        Args:
            x: Input tensor
            instance_ids: Instance ID(s) for normalization, can be a single ID or list of IDs for batch

        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            # Apply normalization if available
            if self.normalizer is not None and instance_ids is not None:
                x = self.normalizer.normalize(x, instance_ids)

            predictions = self.forward(x)

            # Apply denormalization if available
            if self.normalizer is not None and instance_ids is not None:
                predictions = self.normalizer.denormalize(predictions, instance_ids)

            return predictions

    @torch.no_grad()
    def forecast(
        self,
        sequence: torch.Tensor,
        n_steps: int,
        instance_ids: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Generate recursive predictions for n_steps ahead.

        This is a unified approach for all model types, handling
        multi-step prediction by recursively feeding predictions
        back as inputs for future steps.

        Args:
            sequence: Input sequence tensor with shape either:
                      - (sequence_length,) - single sequence, will be treated as batch_size=1
                      - (batch_size, sequence_length) - batch of sequences of the same length
            n_steps: Number of steps to predict
            instance_ids: Instance ID(s) for normalization, can be:
                          - A single integer ID (applied to all sequences)
                          - A list of integer IDs (one per sequence)

        Returns:
            Predictions tensor of shape (batch_size, n_steps)
        """
        self.eval()

        if not isinstance(sequence, torch.Tensor):
            raise TypeError("sequence must be a torch.Tensor")

        if len(sequence.shape) == 1:  # Single sequence without batch dimension
            sequence = sequence.unsqueeze(0)  # Add batch dimension

        sequence = sequence.to(self.device)

        # Handle instance_ids preparation
        if isinstance(instance_ids, int) and sequence.shape[0] > 1:
            # If we have a single instance_id but multiple sequences, duplicate it
            instance_ids = [instance_ids] * sequence.shape[0]
        elif instance_ids is not None and not isinstance(instance_ids, (int, list)):
            raise TypeError(
                "instance_ids must be either an int, a list of ints, or None"
            )

        # Check if list of instance_ids matches batch size
        if isinstance(instance_ids, list) and len(instance_ids) != sequence.shape[0]:
            raise ValueError(
                f"Length of instance_ids list ({len(instance_ids)}) must match batch size ({sequence.shape[0]})"
            )

        # Apply normalization if available
        if self.normalizer is not None and instance_ids is not None:
            sequence = self.normalizer.normalize(sequence, instance_ids)

        if len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(-1)  # Add feature dimension if missing

        batch_size = sequence.shape[0]
        seq_len = sequence.shape[1]

        prediction_length = self.config.get("prediction_length", 1)

        predictions = torch.zeros((batch_size, n_steps), device=self.device)
        current_sequence = sequence.clone()

        steps_done = 0
        while steps_done < n_steps:
            output = self.forward(current_sequence)

            steps_to_use = min(prediction_length, n_steps - steps_done)
            predictions[:, steps_done : steps_done + steps_to_use] = output[
                :, :steps_to_use
            ]

            steps_done += steps_to_use

            if steps_done < n_steps:
                new_values = output[:, :steps_to_use].unsqueeze(-1)

                if seq_len > steps_to_use:
                    current_sequence = torch.cat(
                        [current_sequence[:, steps_to_use:, :], new_values], dim=1
                    )
                else:
                    current_sequence = torch.cat(
                        [current_sequence[:, steps_to_use:, :], new_values], dim=1
                    )[:, -seq_len:, :]

        # Denormalize predictions if needed
        if self.normalizer is not None and instance_ids is not None:
            predictions = self.normalizer.denormalize(predictions, instance_ids)

        return predictions

    def save(self, path: str) -> None:
        """
        Save model weights and normalizer parameters.

        Args:
            path: Path to save the model
        """
        save_dict = {"model_state_dict": self.state_dict(), "config": self.config}

        if self.normalizer is not None:
            save_dict["normalizer"] = self.normalizer.get_params()

        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """
        Load model weights and normalizer parameters.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)

        if "config" in checkpoint:
            self.config = checkpoint["config"]
            if not hasattr(self, "initialized") or not self.initialized:
                self._build_model(self.config)

        self.load_state_dict(checkpoint["model_state_dict"])

        if "normalizer" in checkpoint and self.normalizer is not None:
            self.normalizer.set_params(checkpoint["normalizer"])
