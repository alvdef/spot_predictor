import torch
import os


class CombinedLoss(torch.nn.Module):
    def __init__(self, mse_weight=0.7):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()

    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + (
            1 - self.mse_weight
        ) * self.mae(pred, target)


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints"):
    """
    Saves the model checkpoint to a specified directory. If not best, saved as "last_model.pth".
    Args:
        state (dict): A dictionary containing the model's state_dict, loss, and configuration.
        is_best (bool): A flag indicating whether the current checkpoint is the best model.
        checkpoint_dir (str, optional): The directory where the checkpoint will be saved. Defaults to "checkpoints".
    Returns:
        None
    """
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    file_name = "best_model.pth" if is_best else "last_model.pth"
    file_path = os.path.join(checkpoint_dir, file_name)
    torch.save(state, file_path)


def load_checkpoint(model, checkpoint_dir="checkpoints"):
    """Load the best model checkpoint"""
    best_file = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_file):
        return None, float("inf")

    print(f"\nLoading best model from: {best_file}")
    checkpoint = torch.load(best_file, map_location="cpu")

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint.get("config", None), checkpoint.get("loss", float("inf"))
