from typing import Dict, Optional, Tuple
import os
import torch

from .logging_config import get_logger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


class CheckpointTracker:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint tracker with specified directory."""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.device = get_device()
        self.logger = get_logger(__name__)

    def save_if_best(self, model_state: Dict) -> None:
        """Save checkpoint if metrics have improved."""
        self.save(model_state, is_best=self._is_best(model_state["loss"]))

    def _is_best(self, current_loss: float) -> bool:
        """Determine if current metrics represent best model performance."""
        best_file = os.path.join(self.checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_file):
            return True

        try:
            checkpoint = torch.load(best_file, weights_only=True)
            previous_loss = checkpoint.get("loss", float("inf"))
            return current_loss < previous_loss
        except:
            return True

    def save(self, model_state: Dict, is_best: bool) -> None:
        """Save model checkpoint with specified metrics."""
        file_name = "best_model.pth" if is_best else "last_model.pth"
        file_path = os.path.join(self.checkpoint_dir, file_name)

        torch.save(model_state, file_path)

    def load(self, model: torch.nn.Module) -> Tuple[Optional[Dict], float]:
        """Load best checkpoint if available and compatible."""
        best_file = os.path.join(self.checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_file):
            return None, float("inf")

        try:
            checkpoint = torch.load(
                best_file, map_location=self.device, weights_only=True
            )

            current_state = model.state_dict()
            checkpoint_state = checkpoint["model_state_dict"]

            if not self._architectures_compatible(current_state, checkpoint_state):
                self.logger.warning(
                    "Checkpoint architecture incompatible with current model."
                )
                return checkpoint.get("config", None), float("inf")

            # Load weights if compatible
            model.load_state_dict(checkpoint_state)
            self.logger.info("Successfully loaded checkpoint.")
            return checkpoint.get("config", None), checkpoint.get("loss", float("inf"))

        except Exception as e:
            self.logger.warning(f"Error loading checkpoint - {str(e)}")
            return None, float("inf")

    def _architectures_compatible(
        self, current_state: Dict, checkpoint_state: Dict
    ) -> bool:
        """Check if model architectures are compatible."""
        for key in current_state.keys():
            if key in checkpoint_state:
                if current_state[key].shape != checkpoint_state[key].shape:
                    return False
        return True
