from typing import List
import torch
import torch.nn.functional as F

from .base import LossFunction
from utils import (
    significant_trend_accuracy,
    spot_price_savings,
    perfect_information_savings,
    differentiable_trend_loss,
    calculate_savings_efficiency,
    mean_absolute_percentage_error,
    mse_loss,
    get_logger,
)


class TrendFocusLoss(LossFunction):
    REQUIRED_FIELDS = [
        "mse_weight",
        "significant_threshold",
        "smoothing_factor",
    ]

    def __init__(self, work_dir):
        super().__init__(work_dir)
        self.mse_weight = float(self.config["mse_weight"])
        self.trend_weight = 1 - self.mse_weight
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

        # Standard MSE loss for baseline accuracy using the centralized function
        mse_loss_value = mse_loss(y_pred, y_true)

        # Calculate differences for trend analysis
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]

        # Use the centralized differentiable trend loss function
        trend_loss_value = differentiable_trend_loss(
            pred_diff, true_diff, self.significant_threshold, self.smoothing_factor
        )

        # Combined Loss
        combined_loss = (
            self.mse_weight * mse_loss_value + self.trend_weight * trend_loss_value
        )

        # Calculate metrics for monitoring (non-differentiable, used for reporting only)
        with torch.no_grad():
            # Compute MAPE for reporting
            mape = mean_absolute_percentage_error(y_pred, y_true)

            # Get trend accuracy using the utility function
            sig_trend_acc = significant_trend_accuracy(
                y_pred, y_true, self.significant_threshold
            )

            # Calculate cost savings metrics if we have enough timesteps
            if (
                y_pred.shape[1] > 3
            ):  # Need at least a few timesteps for meaningful savings calc
                cost_savings = spot_price_savings(y_pred, y_true)
                perfect_savings = perfect_information_savings(y_true)

                # Calculate efficiency using the utility function
                savings_efficiency = calculate_savings_efficiency(
                    cost_savings, perfect_savings
                )
            else:
                cost_savings = 0.0
                savings_efficiency = 0.0

        # Return metrics for logging
        metrics = {
            "mse": mse_loss_value.item(),
            "trend_loss": trend_loss_value.item(),
            "mape": mape,
            "sig_trnd_acc": sig_trend_acc,
            "cost_sav": cost_savings,
            "sav_effi": savings_efficiency,
        }

        return combined_loss, metrics

    def get_metric_names(self) -> List[str]:
        """
        Returns the names of metrics this loss function calculates.
        """
        return ["mse", "trend_loss", "mape", "sig_trnd_acc", "cost_sav", "sav_effi"]
