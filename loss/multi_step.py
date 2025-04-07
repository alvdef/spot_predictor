from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossFunction


class MultiStepMSELoss(LossFunction):

    REQUIRED_FIELDS = []

    def __init__(self, work_dir):
        super().__init__(work_dir)

    def forward(self, y_pred, y_true):
        """
        Calculate combined loss

        Args:
            y_pred: Model predictions (batch_size, forecast_steps)
            y_true: Ground truth values (batch_size, forecast_steps)

        Returns:
            tuple: (combined_loss, metrics_dict)
        """
        # Ensure tensors have same shape
        if y_pred.shape != y_true.shape:
            y_pred = y_pred.view(y_true.shape)

        # MSE loss for magnitude accuracy
        mse_loss = F.mse_loss(y_pred, y_true)

        # MAPE loss with improved stability
        epsilon = 1e-8  # For numerical stability

        # Clip values to avoid extreme outliers
        y_true_safe = torch.clamp(y_true.abs(), min=epsilon)
        abs_percentage_error = torch.abs((y_true - y_pred) / y_true_safe)
        abs_percentage_error = torch.clamp(abs_percentage_error, max=10.0)
        mape_loss = torch.mean(abs_percentage_error) * 100

        # Metrics for logging
        metrics = {
            "mse": mse_loss.item(),
            "mape": mape_loss.item(),
        }

        return mse_loss, metrics

    def get_metric_names(self) -> List[str]:
        """
        Returns the names of metrics this loss function calculates.
        """
        return ["mse", "mape"]
