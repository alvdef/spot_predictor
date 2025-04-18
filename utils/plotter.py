import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import pandas as pd
import os
from collections import defaultdict
import math


class ResultPlotter:
    """Plot training results and model predictions."""

    def __init__(self, work_dir: str = "output"):
        """Initialize plotter with output directory."""
        self.output_dir = work_dir
        self.setup_style()

    def setup_style(self):
        """Configure matplotlib style for consistent plots."""
        sns.set_theme()
        sns.set_palette("husl")

    def plot_training_history(
        self, history: Dict[str, List[float]], save: bool = True
    ) -> None:
        """Plot training history including losses, learning rates, gradients, and all available metrics."""
        # Extract all metric names from the history
        metrics = self._extract_available_metrics(history)

        # Always include losses, learning rate, and gradient norm plots
        num_fixed_plots = 3  # Losses, LR, Gradient Norm
        num_metrics = len(metrics)

        # Calculate grid dimensions - first determine how many additional rows we need for metrics
        total_plots = num_fixed_plots + num_metrics
        num_cols = 2  # Use 2 columns
        num_rows = math.ceil(total_plots / num_cols)

        # Create figure with dynamic size based on number of plots
        fig_height = 5 * num_rows
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, fig_height))
        axes = axes.flatten() if num_rows > 1 or num_cols > 1 else [axes]

        # Plot 1: Loss curves (always include)
        self._plot_loss_curves(history, axes[0])

        # Plot 2: Learning rates (always include)
        self._plot_learning_rates(history, axes[1])

        # Plot 3: Gradient norms (always include)
        self._plot_gradient_norms(history, axes[2])

        # Plot the remaining metrics
        for i, metric_name in enumerate(metrics):
            ax_idx = i + num_fixed_plots
            if ax_idx < len(axes):
                self._plot_metric(history, metric_name, axes[ax_idx])

        # Hide any unused axes
        for i in range(total_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/training_history.png", bbox_inches="tight")
        plt.show()

    def _extract_available_metrics(self, history: Dict) -> List[str]:
        """Extract all available metrics from the history dictionary."""
        metrics = []

        # Extract metrics from train_metrics and val_metrics
        if "train_metrics" in history:
            for metric_name in history["train_metrics"].keys():
                # Skip metrics that are plotted separately
                if metric_name not in ("learning_rate", "grad_norm"):
                    metrics.append(metric_name)

        # Check for any metrics at the top level that aren't already handled
        for key in history.keys():
            if (
                key
                not in [
                    "train_loss",
                    "val_loss",
                    "learning_rates",
                    "grad_norms",
                    "epoch_duration",
                    "train_metrics",
                    "val_metrics",
                ]
                and isinstance(history[key], list)
                and len(history[key]) > 0
                and key not in metrics
            ):
                metrics.append(key)

        return metrics

    def _get_metric_data(
        self, history: Dict, metric_name: str, prefix: str = ""
    ) -> np.ndarray:
        """Get metric data with fallback to nested locations."""
        # Try direct access first
        if prefix + metric_name in history and len(history[prefix + metric_name]) > 0:
            return np.array(history[prefix + metric_name])

        # Try nested access under train_metrics/val_metrics
        nested_key = "train_metrics" if prefix == "" else "val_metrics"
        if nested_key in history and metric_name in history[nested_key]:
            return np.array(history[nested_key][metric_name])

        return np.array([])

    def _plot_loss_curves(self, history: Dict, ax: plt.Axes) -> None:
        """Plot train and validation loss curves with outlier handling."""
        train_loss = np.array(history["train_loss"])
        val_loss = np.array(history.get("val_loss", []))

        # Calculate reasonable loss range (excluding outliers)
        if len(train_loss) > 0:
            loss_threshold = np.percentile(train_loss[~np.isnan(train_loss)], 95)
            train_loss[train_loss > loss_threshold] = loss_threshold
            ax.plot(train_loss, label="Training Loss")

        if len(val_loss) > 0 and not all(x is None for x in val_loss):
            # Apply same threshold to validation loss
            val_loss[val_loss > loss_threshold] = loss_threshold
            ax.plot(val_loss, label="Validation Loss")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()
        ax.set_title("Loss Curves")

    def _plot_learning_rates(self, history: Dict, ax: plt.Axes) -> None:
        """Plot learning rates with log scale."""
        if "learning_rates" in history:
            lr_data = np.array(history["learning_rates"])
            valid_lr = lr_data[~np.isnan(lr_data) & (lr_data > 0)]
            if len(valid_lr) > 0:
                min_lr = max(np.min(valid_lr), 1e-10)
                max_lr = np.max(valid_lr)
                ax.plot(lr_data, label="Learning Rate")
                ax.set_xlabel("Step")
                ax.set_ylabel("Learning Rate")
                ax.set_yscale("log")
                ax.set_ylim(min_lr / 10, max_lr * 10)
                ax.grid(True)
                ax.legend()
                ax.set_title("Learning Rate")

    def _plot_gradient_norms(self, history: Dict, ax: plt.Axes) -> None:
        """Plot gradient norms with outlier removal."""
        if "grad_norms" in history:
            grad_norms = np.array(history["grad_norms"])
            valid_norms = grad_norms[~np.isnan(grad_norms) & (grad_norms > 0)]
            if len(valid_norms) > 0:
                norm_threshold = np.percentile(valid_norms, 95)
                grad_norms[grad_norms > norm_threshold] = norm_threshold
                ax.plot(grad_norms, label="Gradient Norm")
                ax.set_xlabel("Step")
                ax.set_ylabel("Gradient Norm")
                ax.grid(True)
                ax.legend()
                ax.set_title("Gradient Norms")

    def _plot_metric(self, history: Dict, metric_name: str, ax: plt.Axes) -> None:
        """Plot a specific metric with both train and validation data if available."""
        train_metric = self._get_metric_data(history, metric_name)
        val_metric = self._get_metric_data(history, metric_name, "val_")

        has_data = False
        if len(train_metric) > 0:
            # Handle outliers for better visualization
            if np.std(train_metric) > np.mean(train_metric) * 10:
                percentile_95 = np.percentile(train_metric[~np.isnan(train_metric)], 95)
                train_metric[train_metric > percentile_95] = percentile_95

            ax.plot(train_metric, label=f"Train {metric_name}")
            has_data = True

        if len(val_metric) > 0:
            # Handle outliers for better visualization
            if np.std(val_metric) > np.mean(val_metric) * 10:
                percentile_95 = np.percentile(val_metric[~np.isnan(val_metric)], 95)
                val_metric[val_metric > percentile_95] = percentile_95

            ax.plot(val_metric, label=f"Validation {metric_name}")
            has_data = True

        if has_data:
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.grid(True)
            ax.legend()
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")

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
