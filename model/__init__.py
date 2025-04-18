import yaml
from .base import Model
from .gru import GRU
from .lstm import LSTM
from .seq2seq import Seq2Seq
from .feature_seq2seq import FeatureSeq2Seq


MODEL_REGISTRY = {
    "GRU": GRU,
    "LSTM": LSTM,
    "Seq2Seq": Seq2Seq,
    "FeatureSeq2Seq": FeatureSeq2Seq,
}


def get_model(work_dir: str, dataset):
    """
    Factory function to create a model instance based on model_type.

    Args:
        work_dir (str): Working directory containing model configuration
        dataset: Optional dataset instance to derive feature sizes from

    Returns:
        Model: An instance of the requested model

    Raises:
        ValueError: If model_type is not in the registry
    """
    with open(f"{work_dir}/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_type = config["model_config"]["model_type"]

    if model_type not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. Available models: {available_models}"
        )

    if "instance_feature_size" not in config["model_config"]:
        config["model_config"]["instance_feature_size"] = dataset.len_instance_features

    if "time_feature_size" not in config["model_config"]:
        config["model_config"]["time_feature_size"] = dataset.len_time_features

    with open(f"{work_dir}/config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    model_class = MODEL_REGISTRY[model_type]
    return model_class(work_dir)


__all__ = ["get_model", "Model"]
