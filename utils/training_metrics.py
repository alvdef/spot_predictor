from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
import json
import torch


@dataclass
class TrainingHistory:
    """Stores training metrics history for visualization and analysis."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    epoch_duration: List[float] = field(default_factory=list)
    mape: List[float] = field(default_factory=list)
    direction_accuracy: List[float] = field(default_factory=list)
    val_mape: List[float] = field(default_factory=list)
    val_direction_accuracy: List[float] = field(default_factory=list)
    mse: List[float] = field(default_factory=list)
    val_mse: List[float] = field(default_factory=list)


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
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_path = os.path.join(self.output_dir, "training_history.json")
        self.metrics_path = os.path.join(self.output_dir, "metrics.json")

    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history.__dict__

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

        # Record training metrics
        if "mape" in train_metrics:
            self._history.mape.append(train_metrics["mape"])
        if "direction" in train_metrics:
            self._history.direction_accuracy.append(train_metrics["direction"])
        if "mse" in train_metrics:
            self._history.mse.append(train_metrics["mse"])

        # Record validation metrics if available
        if val_loss is not None:
            self._history.val_loss.append(val_loss)
            self._best_val_loss = min(self._best_val_loss, val_loss)

            if val_metrics:
                if "mape" in val_metrics:
                    self._history.val_mape.append(val_metrics["mape"])
                if "direction" in val_metrics:
                    self._history.val_direction_accuracy.append(
                        val_metrics["direction"]
                    )
                if "mse" in val_metrics:
                    self._history.val_mse.append(val_metrics["mse"])

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
        display_metrics = {}
        for key, value in metrics.items():
            display_metrics[key] = (
                f"{value.item() if isinstance(value, torch.Tensor) else value:.4f}"
            )

        display_metrics["loss"] = f"{loss.item():.4f}"
        pbar.set_postfix(display_metrics, refresh=False)  # Avoid unnecessary refreshes

    def print_header(self) -> None:
        """Prints the header row for training progress display."""
        print(
            f"{'Epoch':^8} | {'Train Loss':^12} | {'Val Loss':^12} | {'MSE':^8} | {'MAPE':^8} | "
            f"{'Dir Acc':^8} | {'LR':^10} | {'Grad Norm':^10} | {'Duration':^8}"
        )
        print("-" * 115)

    def print_epoch_stats(self) -> None:
        """
        Prints statistics for the current epoch using stored state.
        Uses 'N/A' for missing metrics to maintain consistent output formatting.
        """
        val_str = f"{self._val_loss:.4f}" if self._val_loss is not None else "N/A"

        mse_str = (
            f"{self._train_metrics.get('mse', 0):.2f}"
            if "mse" in self._train_metrics
            else "N/A"
        )
        mape_str = (
            f"{self._train_metrics.get('mape', 0):.2f}"
            if "mape" in self._train_metrics
            else "N/A"
        )
        dir_acc_str = (
            f"{self._train_metrics.get('direction', 0):.2f}"
            if "direction" in self._train_metrics
            else "N/A"
        )

        # Using end='' prevents double newlines when called in loops
        print(
            f"{self._epoch:^8d} | {self._train_loss:^12.4f} | {val_str:^12} | "
            f"{mse_str:^8} | {mape_str:^8} | {dir_acc_str:^8} | "
            f"{self._learning_rate:^10.1e} | {self._grad_norm:^10.1f} | {self._epoch_duration:^8.1f}",
            end="",
        )

        # Record MSE if present - moved from print function to ensure data is captured
        if "mse" in self._train_metrics:
            self._history.mse.append(self._train_metrics["mse"])
        if self._val_metrics and "mse" in self._val_metrics:
            self._history.val_mse.append(self._val_metrics["mse"])

    def print_early_stopping(self) -> None:
        """Prints early stopping notification."""
        print("-" * 100)
        print(
            f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement."
        )

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

            # Include final performance metrics when available
            if self._history.mape:
                metrics["final_mape"] = self._history.mape[-1]
            if self._history.direction_accuracy:
                metrics["final_direction_accuracy"] = self._history.direction_accuracy[
                    -1
                ]
            if self._history.val_mape:
                metrics["final_val_mape"] = self._history.val_mape[-1]
            if self._history.val_direction_accuracy:
                metrics["final_val_direction_accuracy"] = (
                    self._history.val_direction_accuracy[-1]
                )

            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f)

        except Exception as e:
            # Log the error but don't crash the program
            print(f"Error saving metrics data: {str(e)}")
