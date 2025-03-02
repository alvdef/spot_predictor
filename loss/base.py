from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class LossFunction(nn.Module, ABC):
    """
    LossFunction is an abstract base class for loss functions that extends PyTorch's nn.Module.
    It provides a common interface for all loss functions in the project.

    Methods:
        forward(y_pred, y_true):
            Abstract method to be implemented by subclasses to calculate the loss.

        log_metrics(metrics):
            Utility method to format metrics for logging.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate the loss between predictions and ground truth.

        Args:
            y_pred: Model predictions
            y_true: Ground truth values

        Returns:
            tuple: (loss_value, metrics_dict) where metrics_dict contains additional metrics for logging
        """
        pass

    def log_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics for logging.

        Args:
            metrics: Dictionary of metric names and values

        Returns:
            Formatted string of metrics
        """
        return " | ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
