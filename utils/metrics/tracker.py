from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
import json
import torch
from collections import defaultdict

from ..logging_config import get_logger


@dataclass
class TrainingHistory:
    """Stores training metrics history for visualization and analysis."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    epoch_duration: List[float] = field(default_factory=list)
    # Use defaultdict for convenient appending, even if key doesn't exist yet
    train_metrics: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    val_metrics: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class MetricsTracker:
    """
    Tracks training metrics, manages early stopping, saves results, and handles checkpointing.

    Combines robustness and detailed reporting with concise implementation where appropriate.
    """

    output_dir: str
    early_stopping_patience: int = 5

    # Internal state
    _history: TrainingHistory = field(default_factory=TrainingHistory)
    _best_loss: float = float("inf")
    _best_val_loss: float = float("inf")  # Track best raw validation loss separately
    _epochs_no_improve: int = 0

    # Current epoch state (updated at the end of each epoch)
    _epoch: int = 0
    _train_loss: float = float("inf")
    _val_loss: Optional[float] = None
    _train_metrics: Dict[str, float] = field(default_factory=dict)
    _val_metrics: Dict[str, float] = field(default_factory=dict)
    _learning_rate: float = 0.0
    _grad_norm: float = 0.0
    _epoch_duration: float = 0.0
    _metrics_keys: Optional[
        List[str]
    ] = None  # Preserves desired header order for metrics

    def __post_init__(self):
        """Initializes logger, paths, and ensures output directory exists."""
        self.logger = get_logger(__name__)
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_path = os.path.join(self.output_dir, "training_history.json")
        self.metrics_path = os.path.join(self.output_dir, "metrics.json")

    def load(self) -> bool:
        """
        Loads previous training history and metrics from JSON files.

        Handles the first execution case (no files exist) gracefully.

        Returns:
            bool: True if history was successfully loaded, False otherwise
        """
        # If history file doesn't exist, it's likely the first training run
        if not os.path.exists(self.history_path):
            self.logger.info(
                f"No previous training history found at {self.history_path}"
            )
            return False

        try:
            self.logger.info(f"Loading previous training data from {self.history_path}")
            with open(self.history_path, "r") as f:
                state_dict = json.load(f)
                if len(state_dict["train_loss"]) == 0:
                    return False
                self.load_state_dict(state_dict)
            self.logger.info(
                f"Successfully loaded training history from epoch {self._epoch}"
            )
            return True
        except IOError as e:
            self.logger.warning(f"Error reading training files: {e}")
            return False
        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON in training files: {e}")
            return False
        except Exception as e:
            self.logger.warning(
                f"Unexpected error loading training data: {e}", exc_info=True
            )
            return False

    def save(self) -> bool:
        """
        Saves the complete training history and metrics summary to JSON files.

        This method should be called at the end of training or when interrupting
        training to ensure data is persisted.

        Returns:
            bool: True if saving was successful, False if any errors occurred
        """
        success = True

        # Save complete history (useful for plotting/detailed analysis)
        try:
            with open(self.history_path, "w") as f:
                # Use the history property which handles defaultdict conversion
                json.dump(self.history, f, indent=4)
            self.logger.info(f"Saved training history to {self.history_path}")
        except Exception as e:
            self.logger.error(f"Error saving training history: {e}", exc_info=True)
            success = False

        # Save summary metrics (useful for quick comparison/results tables)
        try:
            # Use history property to access potentially converted dicts
            hist_dict = self.history
            # Build summary dictionary with key training metrics
            metrics_summary = {
                "completed_epochs": self._epoch,
                "best_eval_loss": self._best_loss,
                "best_val_loss": self._best_val_loss if hist_dict["val_loss"] else None,
                "final_train_loss": (
                    self._train_loss if hist_dict["train_loss"] else None
                ),
                "final_val_loss": self._val_loss,  # Already None if validation wasn't run
                "total_training_time_seconds": sum(hist_dict["epoch_duration"]),
                "final_learning_rate": (
                    hist_dict["learning_rates"][-1]
                    if hist_dict["learning_rates"]
                    else None
                ),
                "final_grad_norm": (
                    hist_dict["grad_norms"][-1] if hist_dict["grad_norms"] else None
                ),
                **{
                    f"final_train_{k}": v[-1]
                    for k, v in hist_dict["train_metrics"].items()
                    if v
                },
                **{
                    f"final_val_{k}": v[-1]
                    for k, v in hist_dict["val_metrics"].items()
                    if v
                },
            }

            with open(self.metrics_path, "w") as f:
                json.dump(metrics_summary, f, indent=4)
            self.logger.info(f"Saved metrics summary to {self.metrics_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics summary: {e}", exc_info=True)
            success = False

        return success

    # For backwards compatibility
    def load_history(self) -> bool:
        """
        Legacy method for backwards compatibility.
        Use load() instead.
        """
        return self.load()

    # For backwards compatibility
    def save_to_files(self) -> None:
        """
        Legacy method for backwards compatibility.
        Use save() instead.
        """
        self.save()

    @property
    def history(self) -> Dict[str, Any]:
        """Provides a serializable dictionary representation of the training history."""
        history_dict = self._history.__dict__.copy()
        # Ensure defaultdicts are converted to standard dicts for reliable JSON serialization
        history_dict["train_metrics"] = dict(self._history.train_metrics)
        history_dict["val_metrics"] = dict(self._history.val_metrics)
        return history_dict

    @property
    def best_loss(self) -> float:
        """Returns the best evaluation loss observed so far (val_loss if available, else train_loss)."""
        return self._best_loss

    @best_loss.setter
    def best_loss(self, value: float) -> None:
        """Allows setting best_loss externally (e.g., when resuming from checkpoint)."""
        # Note: Resuming logic should typically restore the full state via load_state_dict
        # which also handles _epochs_no_improve and other related fields.
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
        Updates state with metrics from the completed epoch, checks for improvement
        using strict inequality, and determines if early stopping should occur.

        Returns:
            Tuple[bool, bool]: (is_best_model, should_stop_training) based on evaluation loss.
        """
        self._epoch = epoch
        self._train_loss = train_loss
        self._val_loss = val_loss
        self._train_metrics = train_metrics.copy() if train_metrics else {}
        self._val_metrics = val_metrics.copy() if val_metrics else {}
        self._epoch_duration = epoch_duration
        self._learning_rate = self._train_metrics.get("learning_rate", 0.0)
        self._grad_norm = self._train_metrics.get("grad_norm", 0.0)

        # Use local alias for history object for minor brevity
        hist = self._history
        hist.train_loss.append(self._train_loss)
        hist.learning_rates.append(self._learning_rate)
        hist.grad_norms.append(self._grad_norm)
        hist.epoch_duration.append(self._epoch_duration)

        for key, value in self._train_metrics.items():
            # Avoid duplicating metrics that have dedicated history lists
            if key not in ("learning_rate", "grad_norm"):
                hist.train_metrics[key].append(value)

        if self._val_loss is not None:
            hist.val_loss.append(self._val_loss)
            self._best_val_loss = min(
                self._best_val_loss, self._val_loss
            )  # Track best raw val loss
            if self._val_metrics:
                for key, value in self._val_metrics.items():
                    hist.val_metrics[key].append(value)

        # Determine improvement and early stopping based on evaluation loss
        # Prioritize validation loss if available, otherwise use training loss
        current_eval_loss = (
            self._val_loss if self._val_loss is not None else self._train_loss
        )
        # Use strict inequality: loss must decrease to be considered an improvement
        is_best = current_eval_loss < self._best_loss

        if is_best:
            self.logger.debug(
                f"Epoch {epoch}: New best eval loss {current_eval_loss:.6f} (previously {self._best_loss:.6f})"
            )
            self._best_loss = current_eval_loss
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
            self.logger.debug(
                f"Epoch {epoch}: No improvement. Eval loss {current_eval_loss:.6f}, Best loss {self._best_loss:.6f}. Patience {self._epochs_no_improve}/{self.early_stopping_patience}."
            )

        should_stop = self._epochs_no_improve >= self.early_stopping_patience
        return is_best, should_stop

    def compute_batch_average(
        self, total_loss: float, metrics: Dict[str, float], batch_count: int
    ) -> Tuple[float, Dict[str, float]]:
        """Computes average loss and metrics over batches for epoch-level reporting."""
        # Prevent division by zero if training loop somehow provides batch_count=0
        if batch_count == 0:
            return 0.0, {k: 0.0 for k in metrics}
        # Concise calculation using dict comprehension
        return total_loss / batch_count, {
            k: v / batch_count for k, v in metrics.items()
        }

    def track_batch_metrics(
        self,
        metrics: Dict[str, Any],
        accumulated_metrics: Dict[str, float],
    ) -> None:
        """
        Accumulates metrics from a single batch into the provided dictionary.
        Handles both tensor and scalar values, ensuring float accumulation.
        """
        for key, value in metrics.items():
            # Use .item() to extract scalar from tensor, avoiding potential device issues/memory leaks
            current_value = (
                value.item() if isinstance(value, torch.Tensor) else float(value)
            )
            # Use .get() for safe accumulation, initializing to 0.0 if key is new
            accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + current_value

    def update_progress_bar(
        self, pbar: Any, loss: torch.Tensor, metrics: Dict[str, Any]
    ) -> None:
        """
        Updates a progress bar (e.g., tqdm) with current batch loss and metrics.
        Uses concise dictionary unpacking for construction.
        """
        # Use dict unpacking for concise creation of the postfix dictionary
        display_metrics = {
            "loss": f"{loss.item():.6f}",
            # Format other metrics, extracting scalar value if it's a tensor
            **{
                k: f"{v.item() if isinstance(v, torch.Tensor) else v:.6f}"
                for k, v in metrics.items()
            },
        }
        # Avoid unnecessary frequent refreshes which can slow down training
        pbar.set_postfix(display_metrics, refresh=False)

    def print_header(self, metrics_keys: Optional[List[str]] = None) -> None:
        """
        Prints the header row for training progress table display.
        Uses provided metric keys for consistent column order.
        """
        self._store_metric_keys(metrics_keys)

        header_parts = [f"{'Epoch':^8}", f"{'Train Loss':^14}", f"{'Val Loss':^14}"]

        if self._metrics_keys:
            header_parts.extend([f"{metric:^12}" for metric in self._metrics_keys])
        # Only add generic 'Other Metrics' if keys weren't specified AND dynamic metrics actually exist
        elif self._history.train_metrics or self._history.val_metrics:
            header_parts.append(f"{'Other Metrics':^30}")

        header_parts.extend([f"{'LR':^10}", f"{'Duration':^8}"])

        header_str = " | ".join(header_parts)
        self.logger.info(f"{header_str}")
        self.logger.info("-" * len(header_str))

    def print_epoch_stats(self) -> None:
        """
        Prints the statistics row for the most recently completed epoch.
        Uses stored metric keys for column ordering if available. Logs detailed info.
        """
        epoch_str = f"{self._epoch:^8d}"
        train_loss_str = f"{self._train_loss:^14.6f}"
        val_loss_str = (
            f"{self._val_loss:^14.6f}" if self._val_loss is not None else f"{'N/A':^14}"
        )

        # Prepare dynamic metrics for display, excluding those with dedicated columns
        displayable_metrics = self._val_metrics.copy() if self._val_metrics else {}
        for k, v in self._train_metrics.items():
            if k not in displayable_metrics and k not in ("learning_rate", "grad_norm"):
                displayable_metrics[k] = v

        metric_parts = []
        if self._metrics_keys:
            # Use stored keys for order, safely handling missing metrics with .get
            for key in self._metrics_keys:
                value = displayable_metrics.get(key)
                # Format correctly whether value exists or not
                metric_parts.append(
                    f"{value:^12.4f}" if value is not None else f"{'N/A':^12}"
                )
        elif displayable_metrics:
            # Group metrics for the generic column if no keys specified
            metrics_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(displayable_metrics.items())
            )
            metric_parts.append(f"{metrics_str:^30}")  # Fit into the generic column

        lr_str = f"{self._learning_rate:^10.1e}"
        duration_str = f"{self._epoch_duration:^8.1f}s"  # Add 's' unit for clarity

        row_parts = (
            [epoch_str, train_loss_str, val_loss_str]
            + metric_parts
            + [lr_str, duration_str]
        )
        self.logger.info(" | ".join(row_parts))

    def _store_metric_keys(self, metrics_keys: Optional[List[str]]):
        """Stores the desired order of metric keys for consistent display."""
        # Ensure it's stored as a list or None
        self._metrics_keys = list(metrics_keys) if metrics_keys is not None else None

    def print_early_stopping(self) -> None:
        """Prints and logs a detailed notification when early stopping is triggered."""
        message = (
            f"EARLY STOPPING: Patience of {self.early_stopping_patience} epochs "
            f"without improvement reached at epoch {self._epoch}."
            f" Best eval loss was {self._best_loss:.6f}."
        )
        self.logger.warning(message)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restores the tracker's state from a checkpoint dictionary.
        Uses .get() for robustness against missing keys in older state dicts.
        Correctly restores defaultdict behavior for dynamic metrics history.
        """
        history_state = state_dict.get("history", {})

        # Restore TrainingHistory, ensuring dynamic metrics are defaultdicts
        self._history = TrainingHistory(
            train_loss=history_state.get("train_loss", []),
            val_loss=history_state.get("val_loss", []),
            learning_rates=history_state.get("learning_rates", []),
            grad_norms=history_state.get("grad_norms", []),
            epoch_duration=history_state.get("epoch_duration", []),
            # Crucially, restore as defaultdict to maintain behavior
            train_metrics=defaultdict(list, history_state.get("train_metrics", {})),
            val_metrics=defaultdict(list, history_state.get("val_metrics", {})),
        )

        # Restore internal state counters/values using .get with appropriate defaults
        self._best_loss = state_dict.get("best_loss", float("inf"))
        self._best_val_loss = state_dict.get("best_val_loss", float("inf"))
        self._epochs_no_improve = state_dict.get("epochs_no_improve", 0)

        # Restore last epoch's state (less critical if resuming *before* an epoch)
        self._epoch = state_dict.get("epoch", 0)
        self._train_loss = state_dict.get("train_loss", float("inf"))
        self._val_loss = state_dict.get("val_loss", None)
        self._train_metrics = state_dict.get("train_metrics", {})
        self._val_metrics = state_dict.get("val_metrics", {})
        self._learning_rate = state_dict.get("learning_rate", 0.0)
        self._grad_norm = state_dict.get("grad_norm", 0.0)
        self._epoch_duration = state_dict.get("epoch_duration", 0.0)

        # Restore metric keys order using .get for safety
        self._metrics_keys = state_dict.get("metrics_keys", None)

        self.logger.info(
            f"Restored metrics tracker state from epoch {self._epoch}. "
            f"Best loss: {self._best_loss:.6f}, Patience: {self._epochs_no_improve}"
        )
