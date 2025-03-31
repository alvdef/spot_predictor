from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils.config import load_config


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

    REQUIRED_FIELDS: List[str] = []

    def __init__(self, work_dir: str):
        super().__init__()
        self.work_dir = work_dir

        self.config = load_config(
            f"{work_dir}/config.yaml", "loss_config", self.__class__.REQUIRED_FIELDS
        )

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

    def get_metric_names(self) -> List[str]:
        """
        Returns a list of metric names this loss function calculates.
        
        This method should be overridden by subclasses to provide specific metric names.
        Used by MetricsTracker to display the correct metrics in training output.
        
        Returns:
            List of metric names as strings
        """
        # Default implementation returns empty list
        return []
