import optuna
import json
import os
import logging
from datetime import datetime
from torch.utils.data import DataLoader

from model import (
    SpotBiLSTM,
    train_model,
)
from load_spot_dataset import LoadSpotDataset
from spot_dataset import SpotPriceDataset
from utils import get_device

def setup_logger(output_dir):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"optimization_{timestamp}.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('hyperopt')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_model_config(trial: optuna.Trial):
    """Create model configuration from trial parameters"""
    return {
        "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
        "output_scale": 100.0,
        "window_size": trial.suggest_int("window_size", 12, 48, step=4),
        "batch_size": trial.suggest_int("batch_size", 32, 512, step=32),
        "shuffle_buffer": 1000,
        "epochs": 30,
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "mse_weight": trial.suggest_float("mse_weight", 0.5, 0.9),
        "prediction_length": 1
    }

def objective(trial, train_data, val_data, device):
    """Optuna objective function for hyperparameter optimization"""
    logger = logging.getLogger('hyperopt')
    
    # Get configuration for this trial
    config = create_model_config(trial)
    logger.info(f"\nTrial {trial.number} started with parameters:")
    for key, value in config.items():
        logger.info(f"├─ {key}: {value}")
    
    # Create data loaders with current batch size
    train_loader = DataLoader(
        train_data, 
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True
    )
    
    # Add steps_per_epoch to config
    config["steps_per_epoch"] = len(train_loader)
    
    # Initialize model with trial parameters
    model = SpotBiLSTM(config, device)
    
    try:
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            config=config,
            device=device,
            val_loader=val_loader
        )
        
        best_val_loss = min(history["val_loss"])
        logger.info(f"Trial {trial.number} finished with best validation loss: {best_val_loss:.6f}")
        
        # Report intermediate values
        trial.report(best_val_loss, step=config["epochs"])
        
        # Handle pruning
        if trial.should_prune():
            logger.warning(f"Trial {trial.number} pruned.")
            raise optuna.TrialPruned()
            
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

def run_hyperparameter_optimization(
    train_data,
    val_data,
    device,
    n_trials=100,
    study_name="spot_predictor_optimization",
    storage=None
):
    """Run hyperparameter optimization study with improved settings and logging"""
    # Create output directory and setup logger
    output_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir)
    
    logger.info(f"Starting optimization study: {study_name}")
    logger.info(f"Number of trials: {n_trials}")
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=3,
            n_min_trials=20
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
            constant_liar=True
        )
    )
    
    # Add callbacks for better monitoring
    def observation_callback(study, trial):
        if trial.number % 10 == 0:
            logger.info(f"\nTrial {trial.number} summary:")
            logger.info(f"Current best value: {study.best_value:.6f}")
            logger.info("Current best parameters:")
            for k, v in study.best_params.items():
                logger.info(f"  {k}: {v}")
    
    # Run optimization with callback
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, device),
        n_trials=n_trials,
        timeout=None,
        callbacks=[observation_callback],
        catch=(Exception,),
        gc_after_trial=True
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "timestamp": timestamp
    }
    
    # Save best configuration
    best_config = create_model_config(study.best_trial)
    results["best_config"] = best_config
    
    # Save results to JSON
    output_file = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info("\nOptimization Results:")
    logger.info("=" * 50)
    logger.info(f"Best validation loss: {study.best_value:.6f}")
    logger.info("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        logger.info(f"├─ {param}: {value}")
    logger.info(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    lsd = LoadSpotDataset("config.yaml", "data")

    prices_df, instance_info_df = lsd.load_data()
    device = get_device()
    dataset_config = {
        "sequence_length": 28,
        "window_step": 4,
        "batch_size": 32
    }
    train_df, val_df, test_df = lsd.get_training_validation_test_split(prices_df)
    train_dataset = SpotPriceDataset(train_df, dataset_config, device)
    val_dataset = SpotPriceDataset(val_df, dataset_config, device)

    # Run optimization
    results = run_hyperparameter_optimization(
        train_data=train_dataset,  # Your training dataset
        val_data=val_dataset,      # Your validation dataset
        device=device,
        n_trials=100            # Number of trials to run
    )
