from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossFunction
from utils import mean_absolute_percentage_error, mse_loss


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

        # MSE loss for magnitude accuracy using the centralized function
        mse_loss_value = mse_loss(y_pred, y_true)

        # Calculate MAPE using the centralized function
        with torch.no_grad():
            mape_loss = mean_absolute_percentage_error(y_pred, y_true)

        # Metrics for logging
        metrics = {
            "mse": mse_loss_value.item(),
            "mape": mape_loss,
        }

        return mse_loss_value, metrics

    def get_metric_names(self) -> List[str]:
        """
        Returns the names of metrics this loss function calculates.
        """
        return ["mse", "mape"]
