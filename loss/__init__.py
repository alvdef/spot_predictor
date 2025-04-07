import yaml
from .base import LossFunction
from .multi_step import MultiStepMSELoss
from .trend_focus_loss import TrendFocusLoss


LOSS_REGISTRY = {
    "MultiStepMSELoss": MultiStepMSELoss,
    "TrendFocusLoss": TrendFocusLoss,
}


def get_loss(work_dir):
    """
    Factory function to create a loss instance based on loss_type.

    Args:
        work_dir (str): Working directory containing loss configuration

    Returns:
        Loss: An instance of the requested loss

    Raises:
        ValueError: If loss_type is not in the registry
    """
    with open(f"{work_dir}/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        loss_type = config["loss_config"]["loss_type"]

    if loss_type not in LOSS_REGISTRY:
        available_losss = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown loss type: {loss_type}. Available losses: {available_losss}"
        )

    loss_class: LossFunction = LOSS_REGISTRY[loss_type]
    return loss_class(work_dir)


__all__ = ["get_loss", "LossFunction"]
