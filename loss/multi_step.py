import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossFunction


class MultiStepForecastLoss(LossFunction):
    def __init__(self, mse_weight=1.0, mape_weight=0.0, direction_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.mape_weight = mape_weight
        self.direction_weight = direction_weight
        print(
            f"Initialized MultiStepForecastLoss with weights: MSE={self.mse_weight}, MAPE={self.mape_weight}, Direction={self.direction_weight}"
        )

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
        epsilon = 1e-8  # Increased from 1e-6 for better stability

        # Clip values to avoid extreme outliers
        y_true_safe = torch.clamp(y_true.abs(), min=epsilon)

        abs_percentage_error = torch.abs((y_true - y_pred) / y_true_safe)

        # Cap extreme percentage values
        abs_percentage_error = torch.clamp(abs_percentage_error, max=10.0)

        mape_loss = torch.mean(abs_percentage_error) * 100

        # Direction accuracy with handling of flat segments
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]

        # Consider flat segments (very small changes) as matching direction
        flat_mask = torch.abs(true_diff) < epsilon
        direction_match = torch.logical_or(
            (pred_diff * true_diff > 0),  # Same direction
            (flat_mask & (torch.abs(pred_diff) < epsilon)),  # Both flat
        ).float()

        direction_acc = direction_match.mean()
        direction_loss = 1.0 - direction_acc

        # Combine losses
        combined_loss = (
            self.mse_weight * mse_loss
            + self.mape_weight * mape_loss
            + self.direction_weight * direction_loss
        )

        # Metrics for logging
        metrics = {
            "mse": mse_loss.item(),
            "mape": mape_loss.item(),
            "direction": direction_acc.item(),
        }

        return combined_loss, metrics
