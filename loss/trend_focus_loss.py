from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from .base import LossFunction
from utils.trend_metrics import calculate_significant_trend_accuracy


class TrendFocusLoss(LossFunction):
    REQUIRED_FIELDS = [
        "mse_weight",
        "trend_weight",
        "significant_threshold",
        "smoothing_factor",
    ]

    def __init__(self, work_dir):
        super().__init__(work_dir)
        self.mse_weight = self.config["mse_weight"]
        self.trend_weight = self.config["trend_weight"]
        self.significant_threshold = self.config["significant_threshold"]
        self.smoothing_factor = self.config["smoothing_factor"]

        print(
            f"Initialized TrendFocusLoss with weights: "
            f"MSE={self.mse_weight}, Trend={self.trend_weight}"
        )

    def forward(self, y_pred, y_true):
        """
        Calculate combined loss optimized for trend prediction.

        Args:
            y_pred: Model predictions (batch_size, forecast_steps)
            y_true: Ground truth values (batch_size, forecast_steps)

        Returns:
            tuple: (combined_loss, metrics_dict)
        """
        # Ensure tensors have same shape
        if y_pred.shape != y_true.shape:
            y_pred = y_pred.view(y_true.shape)

        # Standard MSE loss for baseline accuracy
        mse_loss = F.mse_loss(y_pred, y_true)

        # Calculate differences for trend analysis
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]

        # Stability constant
        epsilon = 1e-8

        # Use a differentiable direction matching approximation
        direction_match = self._smooth_direction_match(
            pred_diff, true_diff, self.smoothing_factor, epsilon
        )

        # Focus on significant trends
        mean_price = torch.mean(torch.abs(y_true), dim=1, keepdim=True)
        threshold = self.significant_threshold * mean_price

        # Smooth significance weighting
        significance = torch.sigmoid(
            (torch.abs(true_diff) - threshold) * self.smoothing_factor
        )

        # Weight the direction match by significance
        weighted_match = direction_match * significance
        trend_accuracy = weighted_match.sum() / (significance.sum() + epsilon)

        # Trend loss is the inverse of accuracy
        trend_loss = 1.0 - trend_accuracy

        # Combined Loss
        combined_loss = self.mse_weight * mse_loss + self.trend_weight * trend_loss

        # Calculate metrics for monitoring
        y_true_safe = torch.clamp(y_true.abs(), min=epsilon)
        abs_percentage_error = torch.abs((y_true - y_pred) / y_true_safe)
        abs_percentage_error = torch.clamp(abs_percentage_error, max=10.0)
        mape = torch.mean(abs_percentage_error) * 100

        # Calculate the non-differentiable significant trend accuracy for reporting
        # But not for gradient calculation (detach to be safe)
        with torch.no_grad():
            sig_accuracy = calculate_significant_trend_accuracy(
                y_pred.detach(), y_true.detach(), self.significant_threshold
            )

        # Return metrics for logging
        metrics = {
            "mse": mse_loss.item(),
            "trend_acc": trend_accuracy.item(),
            "mape": mape.item(),
            "snf_trend_acc": sig_accuracy,
        }

        return combined_loss, metrics

    def get_metric_names(self) -> List[str]:
        """
        Returns the names of metrics this loss function calculates.
        """
        return ["mse", "trend_acc", "mape", "snf_trend_acc"]

    def _smooth_direction_match(
        self, pred_diff, true_diff, smoothing_factor=10.0, epsilon=1e-8
    ):
        """
        Computes a differentiable approximation of direction matching for use in loss functions.

        Args:
            pred_diff: Predicted price differences
            true_diff: Actual price differences
            smoothing_factor: Controls the sharpness of the approximation
            epsilon: Small constant for numerical stability

        Returns:
            Tensor with direction match scores in range [0, 1]
        """
        # Apply hyperbolic tangent as a smooth approximation to sign
        pred_sign = torch.tanh(smoothing_factor * pred_diff)
        true_sign = torch.tanh(smoothing_factor * true_diff)

        # Product of signs: +1 when directions match, -1 when opposite
        sign_product = pred_sign * true_sign

        # Convert from [-1, 1] range to [0, 1] range
        direction_match = (sign_product + 1) / 2

        # Handle flat cases (near-zero changes) with a smooth transition
        flat_pred = torch.exp(-(pred_diff**2) / (2 * epsilon**2))
        flat_true = torch.exp(-(true_diff**2) / (2 * epsilon**2))
        flat_score = flat_pred * flat_true

        flat_weight = torch.exp(-(true_diff**2) / (2 * (epsilon * 10) ** 2))

        return flat_weight * flat_score + (1 - flat_weight) * direction_match
