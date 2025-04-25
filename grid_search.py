#!/usr/bin/env python3
"""
Grid search automation script for spot_predictor models.

This script performs a grid search by training multiple models with different 
configurations passed via JSON string in command line. It automatically updates 
the config.yaml file, creates the necessary model directories, and launches 
training jobs in parallel.

python grid_search.py --param-grid '{"model_config": {"model_type": ["FeatureSeq2Seq"], "hidden_size": [64, 128]}}' --max-concurrent 3 --delay 5

"""

import os
import yaml
import shutil
import itertools
import subprocess
import argparse
import json
from datetime import datetime
import time

from utils import setup_logging, get_logger
from utils.gpu.gpu_utils import (
    check_gpu_availability,
    setup_gpu_environment,
    get_cuda_device,
    log_gpu_info,
    TORCH_AVAILABLE,
)

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Default concurrent jobs limit
DEFAULT_MAX_CONCURRENT = 2


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

    reg_str = f"{'-'.join(config['dataset_features']['regions'])}"
    fam_str = "".join(
        map(str, config["dataset_features"]["instance_filters"]["instance_family"])
    )
    gen_str = "".join(
        map(str, config["dataset_features"]["instance_filters"]["generation"])
    )
    model_type = config["model_config"]["model_type"].lower()
    timestep = config["sequence_config"]["timestep_hours"]
    pred_days = config["sequence_config"]["model_pred_days"]
    mse_weight = config["loss_config"]["mse_weight"]
    hidden_size = config["model_config"]["hidden_size"]

    model_name = f"{reg_str}-{fam_str}-{gen_str}-{model_type}-{timestep}h-{pred_days}d-{mse_weight}wt-{hidden_size}hd"

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


def start_training_job(model_name, gpu_id=None):
    """
    Start a training job for the specified model configuration.

    Args:
        model_name (str): Name of the model to train
        gpu_id (int, optional): Specific GPU ID to use for training.
                                If None, auto-select based on availability.

    Returns:
        subprocess.Popen: Process object for the running job
    """
    model_dir = f"_models/{model_name}"
    log_file = f"{model_dir}/output.log"

    # Open log file
    log_fd = open(log_file, "w")

    # Set environment variables for GPU utilization
    env = os.environ.copy()

    # Select GPU device based on availability
    if TORCH_AVAILABLE:
        gpu_info = check_gpu_availability()

        if gpu_info["gpu_available"]:
            # If specific GPU requested and available, use it
            if gpu_id is not None and gpu_id < gpu_info["device_count"]:
                device_id = gpu_id
            else:
                # Auto-select based on least utilized GPU (simplified approach)
                device_id = 0  # Default to first GPU

            # Set CUDA device ID for the process
            env["CUDA_VISIBLE_DEVICES"] = str(device_id)
            device = get_cuda_device(device_id)
            logger.info(f"Training {model_name} using {device} (GPU {device_id})")
        else:
            # No GPU available - use CPU
            env["CUDA_VISIBLE_DEVICES"] = ""
            logger.warning(f"Training {model_name} on CPU, no GPU available")
    else:
        # PyTorch not available - use CPU
        env["CUDA_VISIBLE_DEVICES"] = ""

    # Start the training process
    process = subprocess.Popen(
        f"python load_train_evaluate.py",
        shell=True,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Allow process to continue running after script ends
        env=env,  # Pass GPU-enabled environment
    )

    # Log start time and device info
    with open(f"{model_dir}/job_info.txt", "w") as f:
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Process ID: {process.pid}\n")
        f.write(
            f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not set (CPU)')}\n"
        )
        if TORCH_AVAILABLE and gpu_info.get("gpu_available", False):
            gpu_idx = int(env.get("CUDA_VISIBLE_DEVICES", "0"))
            if gpu_idx < len(gpu_info["gpu_info"]):
                gpu_data = gpu_info["gpu_info"][gpu_idx]
                f.write(
                    f"GPU: {gpu_data['name']} ({gpu_data['memory_total']:.2f} GB)\n"
                )

    logger.info(f"Started training job for model {model_name} with PID {process.pid}")
    return process


def run_grid_search(
    param_grid, max_concurrent=DEFAULT_MAX_CONCURRENT, delay_seconds=10
):
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
    gpu_assignments = {}  # Track which GPU is assigned to which process

    # Check for GPU availability
    gpu_info = check_gpu_availability()
    num_gpus = gpu_info["device_count"] if gpu_info["gpu_available"] else 0

    if num_gpus > 0:
        logger.info(f"Grid search will utilize {num_gpus} available GPU(s)")
        # Configure GPU environment for optimal performance
        setup_gpu_environment()
    else:
        logger.info("Grid search will run on CPU only")

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
                        # Free up the GPU if one was assigned
                        if proc.pid in gpu_assignments:
                            logger.info(f"Freeing GPU {gpu_assignments[proc.pid]}")
                            del gpu_assignments[proc.pid]
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

            # Select GPU for this job if available
            gpu_id = None
            if num_gpus > 0:
                # Simple round-robin GPU assignment
                used_gpus = set(gpu_assignments.values())
                for g in range(num_gpus):
                    if g not in used_gpus:
                        gpu_id = g
                        break
                # If all GPUs are in use, pick the one with the least assignments
                if gpu_id is None:
                    gpu_counts = {}
                    for g in gpu_assignments.values():
                        gpu_counts[g] = gpu_counts.get(g, 0) + 1
                    gpu_id = min(gpu_counts, key=lambda x: gpu_counts[x])

            # Start training job with selected GPU
            process = start_training_job(model_name, gpu_id)
            running_processes.append(process)
            completed_models.append(model_name)

            # Track GPU assignment
            if gpu_id is not None:
                gpu_assignments[process.pid] = gpu_id
                logger.info(f"Assigned GPU {gpu_id} to process {process.pid}")

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

        # Generate completion summary
        with open("grid_search_summary.txt", "w") as f:
            f.write(
                f"Grid search completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total models trained: {len(completed_models)}\n\n")
            f.write("Completed models:\n")
            for model in completed_models:
                f.write(f"- {model}\n")


def parse_arguments():
    """
    Parse command line arguments for grid search configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Grid search for spot price prediction models"
    )

    # Opciones generales
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum number of concurrent jobs (default: {DEFAULT_MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=10,
        help="Delay in seconds between launching jobs (default: 10)",
    )

    # JSON con configuración de parámetros
    parser.add_argument(
        "--param-grid",
        type=str,
        required=True,
        help="JSON string with parameter grid configuration",
    )

    return parser.parse_args()


def get_param_grid_from_args(args):
    """
    Get parameter grid configuration from JSON string.

    Args:
        args (argparse.Namespace): Parsed command line arguments

    Returns:
        dict: Parameter grid configuration
    """
    try:
        return json.loads(args.param_grid)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in --param-grid")
        raise


if __name__ == "__main__":
    args = parse_arguments()
    param_grid = get_param_grid_from_args(args)

    setup_gpu_environment()
    log_gpu_info()

    logger.info(f"Running grid search with configuration:")
    for section, params in param_grid.items():
        logger.info(f"  {section}:")
        for key, values in params.items():
            logger.info(f"    {key}: {values}")

    logger.info("Starting grid search process")
    run_grid_search(
        param_grid, max_concurrent=args.max_concurrent, delay_seconds=args.delay
    )
