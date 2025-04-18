from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from dataset import Normalizer
from utils import get_device, load_config, predict_future_time_features


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
        self.time_features = load_config(
            f"{work_dir}/config.yaml", "dataset_config", ["time_features"]
        )["time_features"]

        self._build_model(self.config)
        self.initialized = True

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
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: A tuple containing sequence and additional feature tensors (sequence, instance_features, time_features)
            target: Optional target tensor for models that use teacher forcing

        Returns:
            Output tensor with predictions
        """
        pass

    @torch.no_grad()
    def forecast(
        self,
        sequence_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        n_steps: int,
        instance_ids: List[int],
    ) -> torch.Tensor:
        """
        Generate recursive predictions for n_steps ahead.

        Args:
            sequence_input: A tuple containing (sequence, instance_features, time_features)
            n_steps: Number of steps to predict
            instance_ids: A list of integer IDs (one per sequence)

        Returns:
            Predictions tensor of shape (batch_size, n_steps)
        """
        self.eval()

        sequence, instance_features, time_features = sequence_input

        if not isinstance(sequence, torch.Tensor):
            raise TypeError("sequence must be a torch.Tensor")

        # Add batch dimension if missing
        if len(sequence.shape) == 1:
            sequence = sequence.unsqueeze(0)

        # Validate batch size matches instance_ids
        batch_size = sequence.shape[0]
        if len(instance_ids) != batch_size:
            raise ValueError(
                f"Length of instance_ids ({len(instance_ids)}) must match batch size ({batch_size})"
            )

        seq_len = sequence.shape[1]
        prediction_length = self.config["prediction_length"]
        
        # Initialize output predictions tensor
        predictions = torch.zeros((batch_size, n_steps), device=self.device)
        
        current_sequence = sequence.clone()
        current_time_features = time_features.clone()

        steps_done = 0
        while steps_done < n_steps:
            if self.normalizer:
                current_sequence = self.normalizer.normalize(current_sequence, instance_ids)
                
            model_input = (current_sequence, instance_features, current_time_features)
            output = self.forward(model_input)
            
            if self.normalizer:
                output = self.normalizer.denormalize(output, instance_ids)

            steps_to_use = min(prediction_length, n_steps - steps_done)
            
            predictions[:, steps_done:steps_done + steps_to_use] = output[:, :steps_to_use]
            steps_done += steps_to_use
            
            if steps_done >= n_steps:
                break

            current_sequence = torch.cat(
                [
                    current_sequence[:, steps_to_use:],  # Remove oldest steps
                    output[:, :steps_to_use].unsqueeze(2),  # Add new predictions
                ],
                dim=1,
            )[:, -seq_len:]  # Ensure sequence length remains constant
            
            # Generate time features for the next prediction window
            current_time_features = predict_future_time_features(
                current_time_features,
                self.time_features,
                self.config["timestep_hours"],
                prediction_length,
            )

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
