import os
import torch
from typing import Dict, Optional, Tuple

from utils import get_device


class CheckpointTracker:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint tracker with specified directory."""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.device = get_device()

    def save_if_best(self, model_state: Dict) -> None:
        """Save checkpoint if metrics have improved."""
        self.save(model_state, is_best=self._is_best(model_state["loss"]))

    def _is_best(self, current_loss: float) -> bool:
        """Determine if current metrics represent best model performance."""
        best_file = os.path.join(self.checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_file):
            return True

        try:
            checkpoint = torch.load(best_file)
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
            checkpoint = torch.load(best_file, map_location=self.device)

            # Verify architecture compatibility
            current_state = model.state_dict()
            checkpoint_state = checkpoint["model_state_dict"]

            if not self._architectures_compatible(current_state, checkpoint_state):
                print(
                    "Warning: Checkpoint architecture incompatible with current model."
                )
                return checkpoint.get("config", None), float("inf")

            # Load weights if compatible
            model.load_state_dict(checkpoint_state)
            print("Successfully loaded checkpoint.")
            return checkpoint.get("config", None), checkpoint.get("loss", float("inf"))

        except Exception as e:
            print(f"Warning: Error loading checkpoint - {str(e)}")
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
