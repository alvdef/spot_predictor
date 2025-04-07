import yaml
from datetime import datetime, timezone, date
from typing import Dict, Any, List


def _derive_sequence_parameters(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "sequence_config" not in config:
        return {}

    seq_config = config["sequence_config"]
    derived = {}

    # Calculate timesteps from days
    hours_per_day = 24
    timestep_hours = seq_config.get("timestep_hours", 1)
    steps_per_day = hours_per_day // timestep_hours

    derived["dataset_features"] = {
        "timestep_hours": timestep_hours,
        # Preserve other dataset features parameters
        **(config.get("dataset_features", {})),
    }

    derived["dataset_config"] = {
        "sequence_length": int(
            round(seq_config["sequence_length_days"] * steps_per_day)
        ),
        "window_step": int(round(seq_config["window_step_days"] * steps_per_day)),
        "prediction_length": int(round(seq_config["model_pred_days"] * steps_per_day)),
        # Preserve other dataset config parameters
        **(config.get("dataset_config", {})),
    }

    # Derive model parameters
    model_config = config.get("model_config", {}).copy()
    model_config["prediction_length"] = derived["dataset_config"]["prediction_length"]
    derived["model_config"] = model_config

    # Derive evaluation parameters
    evaluation_days = seq_config.get("evaluation_days", 1)
    derived["evaluate_config"] = {
        "n_timesteps_metrics": int(round(evaluation_days * steps_per_day)),
        "sequence_length": int(
            round(seq_config["sequence_length_days"] * steps_per_day)
        ),
        "window_step": int(round(seq_config["window_step_days"] * steps_per_day)),
        "prediction_length": int(
            round(seq_config["prediction_length_days"] * steps_per_day)
        ),
        # Preserve other evaluation config parameters
        **(config.get("evaluate_config", {})),
    }

    return derived


def load_config(config_path: str, config_key: str, required_fields: list[str]) -> dict:
    """
    Load and validate configuration from a YAML file with sequence parameter derivation.

    This function reads a YAML configuration file, processes date fields, validates
    required fields, and includes derived sequence parameters when appropriate.

    Args:
        - config_path (str): Path to the YAML configuration file
        - config_key (str): Key to extract from the root of the YAML document
        - required_fields (list[str]): List of field names that must be present in the config

    Returns:
        dict: Processed configuration dictionary with derived parameters when applicable
    """
    # Load complete config file first
    with open(config_path, "r") as config_file:
        full_config = yaml.safe_load(config_file)

    # Extract the requested section
    if config_key not in full_config:
        raise ValueError(f"Configuration section '{config_key}' not found")

    config = full_config[config_key]

    # Process date fields
    for key in config:
        if key.endswith("_date"):
            if isinstance(config[key], str):
                config[key] = datetime.strptime(config[key], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            elif isinstance(config[key], date):
                config[key] = datetime.combine(
                    config[key], datetime.min.time()
                ).replace(tzinfo=timezone.utc)

    # Derive parameters for sequence-dependent sections
    if config_key in [
        "dataset_features",
        "dataset_config",
        "model_config",
        "evaluate_config",
    ]:
        derived_configs = _derive_sequence_parameters(full_config)
        if config_key in derived_configs:
            config = {**config, **derived_configs[config_key]}

    # Validate required fields
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    return config


def get_name():
    with open("config.yaml", "r") as config_file:
        return "_models/" + yaml.safe_load(config_file)["model_name"]
