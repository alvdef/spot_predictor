import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import seaborn as sns
import pandas as pd
import os
from collections import defaultdict


class ResultPlotter:
    """Plot training results and model predictions."""

    def __init__(self, output_dir: str = "output"):
        """Initialize plotter with output directory."""
        self.output_dir = output_dir
        self.setup_style()

    def setup_style(self):
        """Configure matplotlib style for consistent plots."""
        plt.style.use("seaborn")
        sns.set_palette("husl")

    def plot_training_history(
        self, history: Dict[str, List[float]], save: bool = True
    ) -> None:
        """Plot training history including losses, learning rates, and gradients."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot losses, excluding extreme initial values
        train_loss = np.array(history["train_loss"])
        val_loss = np.array(history["val_loss"])

        # Calculate reasonable loss range (excluding outliers)
        loss_threshold = np.percentile(train_loss[~np.isnan(train_loss)], 95)
        train_loss[train_loss > loss_threshold] = loss_threshold

        ax1.plot(train_loss, label="Training Loss")
        if not all(x is None for x in val_loss):
            ax1.plot(val_loss, label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        ax1.legend()

        # Plot learning rates with better y-axis range
        lr_data = np.array(history["learning_rates"])
        valid_lr = lr_data[~np.isnan(lr_data) & (lr_data > 0)]
        if len(valid_lr) > 0:
            min_lr = max(np.min(valid_lr), 1e-10)
            max_lr = np.max(valid_lr)
            ax2.plot(lr_data, label="Learning Rate")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Learning Rate")
            ax2.set_yscale("log")
            ax2.set_ylim(min_lr / 10, max_lr * 10)
            ax2.grid(True)
            ax2.legend()

        # Plot gradient norms with outlier removal
        grad_norms = np.array(history["grad_norms"])
        valid_norms = grad_norms[~np.isnan(grad_norms) & (grad_norms > 0)]
        if len(valid_norms) > 0:
            norm_threshold = np.percentile(valid_norms, 95)
            grad_norms[grad_norms > norm_threshold] = norm_threshold
            ax3.plot(grad_norms, label="Gradient Norm")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Gradient Norm")
            ax3.grid(True)
            ax3.legend()

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/training_history.png")
        plt.show()

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Model Predictions",
        save: bool = True,
    ) -> None:
        """Plot actual vs predicted values with correlation analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot of predictions
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            label="Perfect Prediction",
        )
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Actual vs Predicted Values")
        ax1.legend()
        ax1.grid(True)

        # Distribution of errors
        errors = y_pred - y_true
        ax2.hist(errors, bins=50, alpha=0.75)
        ax2.set_xlabel("Prediction Error")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Prediction Errors")
        ax2.grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/predictions.png")
        plt.show()

    def plot_learning_rate_finder(
        self, log_lrs: List[float], losses: List[float], save: bool = True
    ) -> None:
        """Plot learning rate finder results."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(log_lrs, losses)
        ax.set_xlabel("Log Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Rate Finder")
        ax.grid(True)

        if save:
            plt.savefig(f"{self.output_dir}/lr_finder.png")
        plt.show()

    def plot_instance_predictions(
        self,
        df: pd.DataFrame,
        instance_id: int,
        predictions: np.ndarray,
        timestamps,
        save: bool = True,
    ):
        """Plot predictions with context for a specific instance"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        instance_data = df[df["id_instance"] == instance_id]
        ax.plot(
            instance_data["price_timestamp"],
            instance_data["spot_price"],
            label="Historical",
            alpha=0.6,
        )

        # Plot predictions
        ax.plot(timestamps, predictions, label="Forecast", color="red", alpha=0.8)

        # Add instance metadata if available
        if "instance_type" in df.columns:
            instance_type = df[df["id_instance"] == instance_id]["instance_type"].iloc[
                0
            ]
            ax.set_title(f"Instance {instance_id} ({instance_type})")

        ax.set_xlabel("Time")
        ax.set_ylabel("Spot Price")
        ax.grid(True)
        ax.legend()

        if save:
            plt.savefig(
                os.path.join(self.output_dir, f"prediction_instance_{instance_id}.png")
            )
        plt.show()

    def plot_horizon_metrics(
        self, segmented_metrics: Dict, instance_id=None, save: bool = True
    ):
        """
        Plot metrics by forecast horizon/timesteps.
        Shows how model performance changes as predictions go further in time.

        Args:
            segmented_metrics: Dictionary of metrics per instance from Evaluate.evaluate_all()
            instance_id: Optional specific instance to plot. If None, averages across all instances
            save: Whether to save the plot to disk
        """
        if not segmented_metrics:
            raise ValueError("No evaluation results available")

        # Get data to plot
        if instance_id:
            metrics_data = segmented_metrics[instance_id]
        else:
            # Average metrics across all instances
            all_metrics = defaultdict(list)
            for instance_metrics in segmented_metrics.values():
                for timestep_metric in instance_metrics:
                    n_timestep = timestep_metric["n_timestep"]
                    all_metrics[n_timestep].append(timestep_metric)

            metrics_data = []
            for n_timestep, values in sorted(all_metrics.items()):
                metrics_data.append(
                    {
                        "n_timestep": n_timestep,
                        "rmse": np.mean([m["rmse"] for m in values]),
                        "mape": np.mean([m["mape"] for m in values]),
                        "mape_std": np.std([m["mape"] for m in values]),
                        "smape": np.mean([m["smape"] for m in values]),
                        "direction_accuracy": np.mean(
                            [m["direction_accuracy"] for m in values]
                        ),
                    }
                )

        # Create plot
        metrics = ["rmse", "mape", "smape", "direction_accuracy"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for ax, metric in zip(axes, metrics):
            timesteps = [m["n_timestep"] for m in metrics_data]
            values = [m[metric] for m in metrics_data]

            ax.plot(timesteps, values, "o-", label=f"Mean {metric.upper()}")

            if metric == "mape":
                stds = [m["mape_std"] for m in metrics_data]
                ax.fill_between(
                    timesteps,
                    [v - s for v, s in zip(values, stds)],
                    [v + s for v, s in zip(values, stds)],
                    alpha=0.2,
                    label="±1 std",
                )

            ax.set_xlabel("Prediction Steps Ahead")
            ax.set_ylabel(metric.upper())
            ax.grid(True)
            ax.legend()

        title = f"Forecast Metrics by Horizon"
        if instance_id:
            title += f" (Instance {instance_id})"
        plt.suptitle(title)
        plt.tight_layout()

        if save:
            filename = f"horizon_metrics{'_instance_'+str(instance_id) if instance_id else ''}.png"
            plt.savefig(os.path.join(self.output_dir, filename))
        plt.show()
