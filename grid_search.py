#!/usr/bin/env python3
"""
Grid search automation script for spot_predictor models.

This script performs a grid search by training multiple models with different 
configurations. It automatically updates the config.yaml file, creates the 
necessary model directories, and launches training jobs in parallel.
"""

import os
import yaml
import shutil
import itertools
import subprocess
from pathlib import Path
from datetime import datetime
import time

from utils import setup_logging, get_logger


# Initialize logging
setup_logging(
    log_level="INFO",
    log_file="logs/grid_search.log",
)
logger = get_logger(__name__)


def update_config_file(params):
    """
    Update the config.yaml file with the given parameters.

    Args:
        params (dict): Dictionary containing parameters to update in config.yaml

    Returns:
        str: The model name based on the updated configuration
    """
    # Load current config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update configuration with new parameters
    for section, values in params.items():
        if section not in config:
            config[section] = {}

        for key, value in values.items():
            config[section][key] = value

    # Generate a descriptive model name based on key parameters
    model_name = f"{'-'.join(config['dataset_features']['regions'])}"

    if (
        "instance_filters" in config["dataset_features"]
        and "instance_family" in config["dataset_features"]["instance_filters"]
    ):
        model_name += (
            f"-{config['dataset_features']['instance_filters']['instance_family']}"
        )

    if "generation" in config["dataset_features"]["instance_filters"]:
        gen_str = "".join(config["dataset_features"]["instance_filters"]["generation"])
        model_name += f"-{gen_str}"

    timestep = config["sequence_config"]["timestep_hours"]
    pred_days = config["sequence_config"]["model_pred_days"]
    model_type = config["model_config"]["model_type"].lower()

    model_name += f"-{model_type}-{timestep}h-{pred_days}d"

    # Update the model name in the config
    config["model_name"] = model_name

    # Write updated config back to file
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return model_name


def create_model_directory(model_name):
    """
    Create model directory if it doesn't exist.

    Args:
        model_name (str): Name of the model directory to create

    Returns:
        str: Full path to the model directory
    """
    model_dir = f"_models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def start_training_job(model_name):
    """
    Start a training job for the specified model configuration.

    Args:
        model_name (str): Name of the model to train

    Returns:
        subprocess.Popen: Process object for the running job
    """
    model_dir = f"_models/{model_name}"
    log_file = f"{model_dir}/output.log"

    # Open log file
    log_fd = open(log_file, "w")

    # Start the training process
    process = subprocess.Popen(
        f"caffeinate python load_train_evaluate.py",
        shell=True,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Allow process to continue running after script ends
    )

    # Log start time
    with open(f"{model_dir}/job_info.txt", "w") as f:
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Process ID: {process.pid}\n")

    logger.info(f"Started training job for model {model_name} with PID {process.pid}")
    return process


def run_grid_search(param_grid, max_concurrent=2, delay_seconds=10):
    """
    Run a grid search with the specified parameter combinations.

    Args:
        param_grid (dict): Dictionary where keys are config sections and values are
                          dictionaries of parameter names and lists of values
        max_concurrent (int): Maximum number of concurrent training jobs
        delay_seconds (int): Delay in seconds between launching jobs
    """
    # Get all parameter combinations
    sections = []
    values = []

    for section, params in param_grid.items():
        for param, val_list in params.items():
            sections.append(section)
            values.append((param, val_list))

    # Generate all combinations of parameter values
    param_names = [
        f"{section}.{param}" for section, (param, _) in zip(sections, values)
    ]
    param_values = [val_list for _, (_, val_list) in zip(sections, values)]
    combinations = list(itertools.product(*param_values))

    logger.info(f"Starting grid search with {len(combinations)} combinations")
    logger.info(f"Parameters being varied: {', '.join(param_names)}")

    running_processes = []
    completed_models = []

    # Create a backup of the original config.yaml
    shutil.copy2("config.yaml", "config.yaml.bak")
    logger.info("Created backup of original config.yaml")

    try:
        for i, combination in enumerate(combinations):
            # Wait if we have reached the maximum number of concurrent jobs
            while len(running_processes) >= max_concurrent:
                # Check if any jobs have finished
                still_running = []
                for proc in running_processes:
                    if proc.poll() is None:  # Process still running
                        still_running.append(proc)
                    else:
                        logger.info(f"Process {proc.pid} has completed")

                running_processes = still_running

                if len(running_processes) >= max_concurrent:
                    logger.info(
                        f"Waiting for a job to complete ({len(running_processes)}/{max_concurrent} running)"
                    )
                    time.sleep(30)  # Wait 30 seconds before checking again

            # Build parameter dictionary for this combination
            params = {}
            for j, ((section, (param, _)), value) in enumerate(
                zip(zip(sections, values), combination)
            ):
                if section not in params:
                    params[section] = {}
                params[section][param] = value

            # Update config and get model name
            model_name = update_config_file(params)
            logger.info(f"Running combination {i+1}/{len(combinations)}: {model_name}")

            # Create model directory
            model_dir = create_model_directory(model_name)

            # Start training job
            process = start_training_job(model_name)
            running_processes.append(process)
            completed_models.append(model_name)

            # Delay before starting next job
            logger.info(f"Waiting {delay_seconds} seconds before starting next job...")
            time.sleep(delay_seconds)

    except Exception as e:
        logger.error(f"Grid search encountered an error: {str(e)}", exc_info=True)
        raise
    finally:
        # Restore original config file when done or if interrupted
        if os.path.exists("config.yaml.bak"):
            shutil.move("config.yaml.bak", "config.yaml")
            logger.info("Restored original config.yaml")

        logger.info("Grid search completed")
        for model in completed_models:
            logger.info(f"- Completed model: {model}")


if __name__ == "__main__":
    # Define parameter grid
    param_grid = {
        "sequence_config": {
            "timestep_hours": [1, 3, 4, 6, 8, 12],
            # "model_pred_days": [1, 2, 5]
        },
        "model_config": {
            "model_type": ["Seq2Seq", "LSTM"],
            # "hidden_size": [60, 120]
        },
    }

    # Run grid search with max 2 concurrent jobs and 10 second delay between jobs
    logger.info("Starting grid search process")
    run_grid_search(param_grid, max_concurrent=2, delay_seconds=10)
