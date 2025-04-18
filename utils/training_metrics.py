from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
import json
import torch
from collections import defaultdict

from .logging_config import get_logger


@dataclass
class TrainingHistory:
    """Stores training metrics history for visualization and analysis."""

    # Basic metrics that are always tracked
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    epoch_duration: List[float] = field(default_factory=list)

    # Dynamic metrics storage
    train_metrics: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    val_metrics: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class MetricsTracker:
    """
    Tracks and manages training metrics, handles early stopping decisions,
    and provides reporting functionality.
    """

    output_dir: str
    early_stopping_patience: int = 5

    # Internal state
    _history: TrainingHistory = field(default_factory=TrainingHistory)
    _best_loss: float = float("inf")
    _best_val_loss: float = float("inf")
    _epochs_no_improve: int = 0

    # Current epoch state
    _epoch: int = 0
    _train_loss: float = float("inf")
    _val_loss: Optional[float] = None
    _train_metrics: Dict[str, float] = field(default_factory=dict)
    _val_metrics: Dict[str, float] = field(default_factory=dict)
    _learning_rate: float = 0.0
    _grad_norm: float = 0.0
    _epoch_duration: float = 0.0

    def __post_init__(self):
        # Create output directory to ensure files can be saved later
        self.logger = get_logger(__name__)
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_path = os.path.join(self.output_dir, "training_history.json")
        self.metrics_path = os.path.join(self.output_dir, "metrics.json")

    @property
    def history(self) -> Dict[str, Any]:
        # Convert defaultdicts to regular dicts for serialization
        history_dict = self._history.__dict__.copy()
        history_dict["train_metrics"] = dict(self._history.train_metrics)
        history_dict["val_metrics"] = dict(self._history.val_metrics)
        return history_dict

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @best_loss.setter
    def best_loss(self, value: float) -> None:
        # Allow setting best_loss externally for checkpoint resumption
        self._best_loss = value

    def update_epoch_state(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: Optional[float] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        epoch_duration: float = 0.0,
    ) -> Tuple[bool, bool]:
        """
        Updates internal state with metrics from the current epoch and determines
        if early stopping should be triggered.

        Returns:
            Tuple[bool, bool]: (is_best_model, should_stop_training)
        """
        self._epoch = epoch
        self._train_loss = train_loss
        self._val_loss = val_loss
        self._train_metrics = train_metrics.copy() if train_metrics else {}
        self._val_metrics = val_metrics.copy() if val_metrics else {}
        self._learning_rate = train_metrics.get("learning_rate", 0.0)
        self._grad_norm = train_metrics.get("grad_norm", 0.0)
        self._epoch_duration = epoch_duration

        # Update history records
        self._history.train_loss.append(train_loss)
        self._history.learning_rates.append(self._learning_rate)
        self._history.grad_norms.append(self._grad_norm)
        self._history.epoch_duration.append(epoch_duration)

        # Record all training metrics dynamically
        for key, value in train_metrics.items():
            if (
                key != "learning_rate" and key != "grad_norm"
            ):  # Already tracked separately
                self._history.train_metrics[key].append(value)

        # Record validation metrics if available
        if val_loss is not None:
            self._history.val_loss.append(val_loss)
            self._best_val_loss = min(self._best_val_loss, val_loss)

            if val_metrics:
                for key, value in val_metrics.items():
                    self._history.val_metrics[key].append(value)

        # Use validation loss for early stopping if available, otherwise use training loss
        current_eval_loss = val_loss if val_loss is not None else train_loss
        is_best = False
        if current_eval_loss <= self._best_loss:
            self._best_loss = current_eval_loss
            is_best = True
            self._epochs_no_improve = 0  # Reset counter when we find a better model
        else:
            self._epochs_no_improve += 1  # Increment counter when no improvement

        # Signal early stopping if patience threshold is reached
        should_stop = self._epochs_no_improve >= self.early_stopping_patience

        return is_best, should_stop

    def compute_batch_average(
        self, total_loss: float, metrics: Dict[str, float], batch_count: int
    ) -> Tuple[float, Dict[str, float]]:
        """Computes per-batch averages to normalize metrics at epoch end."""
        avg_loss = total_loss / batch_count
        avg_metrics = {k: v / batch_count for k, v in metrics.items()}

        return avg_loss, avg_metrics

    def track_batch_metrics(
        self,
        metrics: Dict[str, Any],
        accumulated_metrics: Dict[str, float],
    ) -> None:
        """
        Accumulates metrics from a batch into the provided dictionary.
        Handles both tensor and scalar values for flexibility.
        """
        for key, value in metrics.items():
            if key not in accumulated_metrics:
                accumulated_metrics[key] = 0.0
            # Extract item from tensor if needed
            accumulated_metrics[key] += (
                value.item() if isinstance(value, torch.Tensor) else value
            )

    def update_progress_bar(
        self, pbar: Any, loss: torch.Tensor, metrics: Dict[str, Any]
    ) -> None:
        """
        Updates progress bar with current batch metrics for real-time feedback during training.
        Extracts scalar values from tensors to avoid CUDA synchronization issues.
        """
        # Create an ordered dictionary with loss first, then other metrics
        display_metrics = {}

        # Add loss first so it appears at the beginning
        display_metrics["loss"] = f"{loss.item():.6f}"

        # Then add remaining metrics
        for key, value in metrics.items():
            display_metrics[key] = (
                f"{value.item() if isinstance(value, torch.Tensor) else value:.6f}"
            )

        pbar.set_postfix(display_metrics, refresh=False)  # Avoid unnecessary refreshes

    def print_header(self, metrics_keys=None) -> None:
        """
        Prints the header row for training progress display with dynamic metric names.

        Args:
            metrics_keys: Optional list of metric names to display in header
        """
        # Store keys for later use in print_epoch_stats
        if metrics_keys:
            self._store_metric_keys(metrics_keys)

        # Create columns for basic metrics
        header = [f"{'Epoch':^8}", f"{'Train Loss':^14}", f"{'Val Loss':^14}"]

        # Add dynamic metric columns
        if metrics_keys:
            for metric in metrics_keys:
                header.append(f"{metric:^12}")
        else:
            header.append(f"{'Metrics':^30}")

        # Add remaining columns
        header.extend([f"{'LR':^10}", f"{'Duration':^8}"])

        # Print the header row
        header_str = " | ".join(header)
        print(header_str)
        self.logger.info(f"Training progress tracking started:\n{header_str}")

        # Calculate total width and print the separator line
        total_width = (
            sum(len(col) for col in header) + (len(header) - 1) * 3
        )  # 3 spaces for " | "
        separator = "-" * total_width
        print(separator)

    def print_epoch_stats(self) -> None:
        """
        Prints statistics for the current epoch using stored state.
        Uses 'N/A' for missing metrics to maintain consistent output formatting.
        """
        # Format base values
        epoch_str = f"{self._epoch:^8d}"
        train_loss_str = f"{self._train_loss:^14.6f}"
        val_loss_str = (
            f"{self._val_loss:^14.6f}" if self._val_loss is not None else f"{'N/A':^14}"
        )

        # Format metrics
        metric_parts = []
        relevant_metrics = {
            k: v
            for k, v in self._train_metrics.items()
            if k not in ("learning_rate", "grad_norm")
        }

        # Get ordered metric values that match the header
        if hasattr(self, "_metrics_keys") and self._metrics_keys:
            for key in self._metrics_keys:
                if key in relevant_metrics:
                    metric_parts.append(f"{relevant_metrics[key]:^12.4f}")
                else:
                    metric_parts.append(f"{'N/A':^12}")
        # Fall back to alphabetical display if no keys defined
        elif relevant_metrics:
            for key, value in sorted(relevant_metrics.items()):
                metric_parts.append(f"{value:^12.4f}")
        else:
            metric_parts.append(f"{'N/A':^30}")

        # Format learning rate and duration
        lr_str = f"{self._learning_rate:^10.1e}"
        duration_str = f"{self._epoch_duration:^8.1f}"

        # Combine all parts and print
        row_parts = (
            [epoch_str, train_loss_str, val_loss_str]
            + metric_parts
            + [lr_str, duration_str]
        )
        row_str = " | ".join(row_parts)
        print(row_str)

        # Also log to file with more concise format for log readability
        log_message = f"Epoch {self._epoch}: train={self._train_loss:.6f}"
        if self._val_loss is not None:
            log_message += f", val={self._val_loss:.6f}"
        if relevant_metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in relevant_metrics.items())
            log_message += f", {metrics_str}"
        log_message += (
            f", lr={self._learning_rate:.1e}, time={self._epoch_duration:.1f}s"
        )
        self.logger.info(log_message)

    # Store metric keys for consistent display order
    def _store_metric_keys(self, metrics_keys):
        """Store metric keys to ensure consistent order in header and stats output."""
        self._metrics_keys = metrics_keys

    def print_early_stopping(self) -> None:
        """Prints early stopping notification."""
        separator = "-" * 100
        message = f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement."
        print(separator)
        print(message)
        self.logger.warning(message)

    def save_to_files(self) -> None:
        """
        Saves training history and final metrics to JSON files.
        Uses try-except to ensure training isn't interrupted if file saving fails.
        """
        try:
            # Save complete training history for detailed analysis
            with open(self.history_path, "w") as f:
                json.dump(self.history, f)

            # Save summary metrics for quick reference and model comparison
            metrics = {
                "best_loss": self._best_loss,
                "final_train_loss": self._train_loss,
                "final_val_loss": self._val_loss,
                "training_time": sum(self._history.epoch_duration),
                "completed_epochs": len(self._history.train_loss),
            }

            # Include final performance metrics dynamically
            for key, values in self._history.train_metrics.items():
                if values:
                    metrics[f"final_{key}"] = values[-1]

            for key, values in self._history.val_metrics.items():
                if values:
                    metrics[f"final_val_{key}"] = values[-1]

            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f)

            self.logger.info(
                f"Saved metrics to {self.metrics_path} and training history to {self.history_path}"
            )

        except Exception as e:
            # Log the error but don't crash the program
            error_message = f"Error saving metrics data: {str(e)}"
            print(error_message)
            self.logger.error(error_message, exc_info=True)
