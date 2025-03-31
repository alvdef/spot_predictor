from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from dataset import Normalizer
from utils import get_device, load_config


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

    REQUIRED_FIELDS: List[str] = []

    def __init__(self, work_dir: str):
        super().__init__()
        self.device = get_device()
        self.normalizer: Optional[Normalizer] = None
        self.work_dir = work_dir

        self.config = load_config(
            f"{work_dir}/config.yaml", "model_config", self.__class__.REQUIRED_FIELDS
        )
        self._build_model(self.config)
        self.initialized = True
        print(f"Initialized {self.__class__.__name__}")

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
    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Either:
               - A sequence tensor
               - A tuple (sequence, features) where:
                 sequence: Input tensor of shape (batch_size, seq_len, input_size)
                 features: Additional features tensor of shape (batch_size, feature_size)
            target: Optional target tensor for models that use teacher forcing

        Returns:
            Output tensor
        """
        pass

    @torch.no_grad()
    def predict(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        instance_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Make predictions with a single forward pass (no recursion).

        Args:
            x: Either:
               - A sequence tensor
               - A tuple (sequence, features) where:
                 sequence: Input tensor
                 features: Additional features tensor
            instance_ids: Instance ID(s) for normalization

        Returns:
            Predictions tensor
        """
        self.eval()
        # Handle sequence and features separately for normalization
        if isinstance(x, tuple):
            sequence, features = x
            if self.normalizer is not None and instance_ids is not None:
                sequence = self.normalizer.normalize(sequence, instance_ids)
            x = (sequence, features)
        else:
            if self.normalizer is not None and instance_ids is not None:
                x = self.normalizer.normalize(x, instance_ids)

        predictions = self.forward(x)

        if self.normalizer is not None and instance_ids is not None:
            predictions = self.normalizer.denormalize(predictions, instance_ids)

        return predictions

    @torch.no_grad()
    def forecast(
        self,
        sequence: torch.Tensor,
        n_steps: int,
        instance_ids: Optional[List[int]] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate recursive predictions for n_steps ahead by using predict.

        Args:
            sequence: Input sequence tensor with shape either:
                      - (sequence_length,) - single sequence, will be treated as batch_size=1
                      - (batch_size, sequence_length) - batch of sequences of the same length
            n_steps: Number of steps to predict
            instance_ids: Instance ID(s) for normalization, can be:
                          - A single integer ID (applied to all sequences)
                          - A list of integer IDs (one per sequence)
            features: Additional features tensor of shape (batch_size, feature_size)

        Returns:
            Predictions tensor of shape (batch_size, n_steps)
        """
        self.eval()

        if not isinstance(sequence, torch.Tensor):
            raise TypeError("sequence must be a torch.Tensor")

        if instance_ids is not None and not isinstance(instance_ids, list):
            raise TypeError(
                "instance_ids must be either an int, a list of ints, or None"
            )

        if isinstance(instance_ids, list) and len(instance_ids) != sequence.shape[0]:
            raise ValueError(
                f"Length of instance_ids list ({len(instance_ids)}) must match batch size ({sequence.shape[0]})"
            )

        if len(sequence.shape) == 1:  # Single sequence without batch dimension
            sequence = sequence.unsqueeze(0)  # Add batch dimension

        batch_size = sequence.shape[0]
        seq_len = sequence.shape[1]
        prediction_length = self.config.get("prediction_length", 1)

        predictions = torch.zeros((batch_size, n_steps), device=self.device)
        current_sequence = sequence.clone()

        steps_done = 0
        while steps_done < n_steps:
            # Prepare input for predict method
            model_input = (
                (current_sequence, features)
                if features is not None
                else current_sequence
            )

            output = self.predict(model_input, instance_ids)

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

        return predictions

    def save(self) -> None:
        """
        Save model weights and normalizer parameters.

        Args:
            path: Path to save the model
        """
        save_dict = {"model_state_dict": self.state_dict(), "config": self.config}

        if self.normalizer is not None:
            save_dict["normalizer"] = self.normalizer.get_params()

        path = self.work_dir + "/model.pth"
        torch.save(save_dict, path)

    def load(self) -> None:
        """
        Load model weights and normalizer parameters.

        Args:
            path: Path to load the model from
        """
        path = self.work_dir + "/model.pth"
        checkpoint = torch.load(path, map_location=self.device)
        if "config" in checkpoint:
            self.config = checkpoint["config"]
            if not hasattr(self, "initialized") or not self.initialized:
                self._build_model(self.config)

        self.load_state_dict(checkpoint["model_state_dict"])

        if "normalizer" in checkpoint:
            self.normalizer = Normalizer(self.device)
            self.normalizer.set_params(checkpoint["normalizer"])
