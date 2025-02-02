from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import json


@dataclass
class LossState:
    best_loss: float = float("inf")
    best_val_loss: float = float("inf")
    current_val_loss: Optional[float] = None
    current_train_loss: float = float("inf")


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    epoch_duration: List[float] = field(default_factory=list)


@dataclass
class MetricsTracker:
    output_dir: str
    _history: TrainingHistory = field(default_factory=TrainingHistory)
    _loss: LossState = field(default_factory=LossState)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_path = os.path.join(self.output_dir, "training_history.json")
        self.metrics_path = os.path.join(self.output_dir, "metrics.json")

    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history.__dict__

    @property
    def current_loss(self) -> float:
        return self._loss.current_train_loss

    @property
    def best_loss(self) -> float:
        return self._loss.best_loss

    @best_loss.setter
    def best_loss(self, value: float) -> None:
        self._loss.best_loss = value

    @property
    def current_val_loss(self) -> Optional[float]:
        return self._loss.current_val_loss

    def update_history(
        self,
        train_loss: float,
        learning_rate: float,
        grad_norm: float,
        epoch_duration: float,
        val_loss: Optional[float] = None,
    ) -> None:
        # Update history
        self._history.train_loss.append(train_loss)
        self._history.learning_rates.append(learning_rate)
        self._history.grad_norms.append(grad_norm)
        self._history.epoch_duration.append(epoch_duration)

        # Update loss tracking
        self._loss.current_train_loss = train_loss
        if val_loss is not None:
            self._history.val_loss.append(val_loss)
            self._loss.current_val_loss = val_loss
            self._loss.best_val_loss = min(self._loss.best_val_loss, val_loss)

        # Update best overall loss
        current_best = val_loss if val_loss is not None else train_loss
        self._loss.best_loss = min(self._loss.best_loss, current_best)

    def save_to_files(self) -> None:
        try:
            # Save training history
            with open(self.history_path, "w") as f:
                json.dump(self.history, f)

            # Save final metrics
            metrics = {
                "best_loss": self.best_loss,
                "final_train_loss": self.current_loss,
                "final_val_loss": self.current_val_loss,
                "training_time": sum(self._history.epoch_duration),
                "completed_epochs": len(self._history.train_loss),
            }
            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f)

        except Exception as e:
            print(f"Error saving metrics data: {str(e)}")
