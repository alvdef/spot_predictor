from typing import List
import torch
import torch.nn.functional as F

from .base import LossFunction
from utils import (
    calculate_significant_trend_accuracy,
    calculate_spot_price_savings,
    calculate_perfect_information_savings,
    get_logger,
)


class TrendFocusLoss(LossFunction):
    REQUIRED_FIELDS = [
        "mse_weight",
        "trend_weight",
        "significant_threshold",
        "smoothing_factor",
    ]

    def __init__(self, work_dir):
        super().__init__(work_dir)
        self.mse_weight = float(self.config["mse_weight"])
        self.trend_weight = float(self.config["trend_weight"])
        self.significant_threshold = float(self.config["significant_threshold"])
        self.smoothing_factor = float(self.config["smoothing_factor"])
        self.logger = get_logger(__name__)

        self.logger.info(
            f"Initialized TrendFocusLoss with weights: "
            f"MSE={self.mse_weight}, Trend={self.trend_weight}, "
            f"Threshold={self.significant_threshold}"
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

        # Differentiable trend direction loss
        trend_loss = self._differentiable_trend_loss(
            pred_diff, true_diff, self.significant_threshold, self.smoothing_factor
        )

        # Combined Loss
        combined_loss = self.mse_weight * mse_loss + self.trend_weight * trend_loss

        # Calculate metrics for monitoring (non-differentiable, used for reporting only)
        # Using the utility functions from trend_metrics.py
        with torch.no_grad():
            # Compute MAPE for reporting
            epsilon = 1e-8
            y_true_safe = torch.clamp(y_true.abs(), min=epsilon)
            abs_percentage_error = torch.abs((y_true - y_pred) / y_true_safe)
            abs_percentage_error = torch.clamp(abs_percentage_error, max=10.0)
            mape = torch.mean(abs_percentage_error) * 100

            # Get trend accuracy using the utility function
            sig_trend_acc = calculate_significant_trend_accuracy(
                y_pred, y_true, self.significant_threshold
            )

            # Calculate cost savings metrics if we have enough timesteps
            if (
                y_pred.shape[1] > 3
            ):  # Need at least a few timesteps for meaningful savings calc
                cost_savings = calculate_spot_price_savings(y_pred, y_true)
                perfect_savings = calculate_perfect_information_savings(y_true)

                # Calculate efficiency (how close to perfect our savings are)
                if perfect_savings > 0:
                    savings_efficiency = (cost_savings / perfect_savings) * 100
                else:
                    savings_efficiency = 100.0 if cost_savings == 0 else 0.0
            else:
                cost_savings = 0.0
                savings_efficiency = 0.0

        # Return metrics for logging
        metrics = {
            "mse": mse_loss.item(),
            "trend_loss": trend_loss.item(),
            "mape": mape.item(),
            "sig_trnd_acc": sig_trend_acc,
            "cost_sav": cost_savings if "cost_sav" in locals() else 0.0,
            "sav_effi": (
                savings_efficiency if "sav_effi" in locals() else 0.0
            ),
        }

        return combined_loss, metrics

    def _differentiable_trend_loss(
        self, pred_diff, true_diff, significance_threshold=0.02, smoothing_factor=10.0
    ):
        """
        Computes a differentiable loss for trend prediction that mimics significant_trend_accuracy
        but maintains gradient flow for training.

        This approximates the calculate_significant_trend_accuracy function while being
        fully differentiable for gradient-based optimization.

        Args:
            pred_diff: Predicted price differences (batch_size, timesteps-1)
            true_diff: Actual price differences (batch_size, timesteps-1)
            significance_threshold: Threshold for significant changes
            smoothing_factor: Controls the sharpness of the approximation

        Returns:
            Differentiable trend loss tensor
        """
        # Stability constant
        epsilon = 1e-8

        # Compute significance weights with a smooth transition at the threshold
        # Higher values have weight closer to 1, lower values closer to 0
        significance_magnitude = torch.abs(true_diff)
        mean_price = torch.mean(torch.abs(true_diff), dim=1, keepdim=True)
        threshold = significance_threshold * mean_price

        # Sigmoid gives a smooth transition from 0 to 1 around the threshold
        significance_weight = torch.sigmoid(
            (significance_magnitude - threshold) * smoothing_factor
        )

        # Get the signs of the differences (direction of change)
        # Using tanh for a differentiable approximation of the sign function
        pred_sign = torch.tanh(smoothing_factor * pred_diff)
        true_sign = torch.tanh(smoothing_factor * true_diff)

        # Compute direction matching score: 1 when signs match, 0 when opposite
        # (true_sign * pred_sign + 1) / 2 maps from [-1,1] to [0,1]
        direction_match = (true_sign * pred_sign + 1) / 2

        # Apply significance weights - only significant changes affect the loss
        weighted_direction_match = direction_match * significance_weight

        # Normalize by the sum of significance weights to get an accuracy-like score
        significance_sum = torch.sum(significance_weight) + epsilon
        trend_accuracy = torch.sum(weighted_direction_match) / significance_sum

        # Loss is 1 - accuracy (we want to maximize accuracy -> minimize loss)
        trend_loss = 1.0 - trend_accuracy

        return trend_loss

    def get_metric_names(self) -> List[str]:
        """
        Returns the names of metrics this loss function calculates.
        """
        return ["mse", "trend_loss", "mape", "sig_trnd_acc", "cost_sav", "sav_effi"]
