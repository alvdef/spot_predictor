import torch
import yaml
from datetime import datetime, timezone, date
from typing import Dict, Any, List


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def _derive_sequence_parameters(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    derived = {}
    if "sequence_config" not in config:
        raise KeyError("Missing required 'sequence_config' section in configuration")

    seq_config = config["sequence_config"]

    hours_per_day = 24
    timestep_hours = seq_config["timestep_hours"]
    steps_per_day = hours_per_day // timestep_hours

    derived = {
        "timestep_hours": timestep_hours,
        "sequence_length": round(seq_config["sequence_length_days"] * steps_per_day),
        "window_step": round(seq_config["window_step_days"] * steps_per_day),
        "tr_prediction_length": round(seq_config["model_pred_days"] * steps_per_day),
        "n_timesteps_metrics": round(seq_config["evaluation_days"] * steps_per_day),
        "prediction_length": round(
            seq_config["prediction_length_days"] * steps_per_day
        ),
    }

    return derived


def load_config(config_path: str, config_key: str, required_fields: list[str]) -> dict:
    """
    Load and validate configuration from a YAML file with sequence parameter derivation.

    This function reads a YAML configuration file, processes date fields, validates
    required fields, and includes derived sequence parameters.

    Args:
        - config_path (str): Path to the YAML configuration file
        - config_key (str): Key to extract from the root of the YAML document
        - required_fields (list[str], optional): List of field names that must be present in the config

    Returns:
        dict: Processed configuration dictionary with derived parameters
    """
    if required_fields is None:
        required_fields = []

    try:
        with open(config_path, "r") as config_file:
            full_config = yaml.safe_load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

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

    derived_configs = _derive_sequence_parameters(full_config)
    config = {**config, **derived_configs}

    # Validate required fields
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    return config


def get_name():
    with open("config.yaml", "r") as config_file:
        return "_models/" + yaml.safe_load(config_file)["model_name"]
