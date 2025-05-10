from datetime import datetime
import json
import os

from model import get_model
from procedures import Training, Evaluate
from dataset import SpotDataset, LoadSpotDataset
from utils import (
    get_name,
    ResultPlotter,
    setup_logging,
    get_logger,
    check_gpu_availability,
    setup_gpu_environment,
    log_gpu_info,
)

# Initialize logging system for the entire application
setup_logging()
logger = get_logger(__name__)

logger.info("Starting spot price prediction pipeline")

# Setup GPU environment for G4DN instances
gpu_info = check_gpu_availability()
if gpu_info["gpu_available"]:
    logger.info(f"GPU detected: {gpu_info['device_count']} available")
    for gpu in gpu_info["gpu_info"]:
        logger.info(
            f"Using GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']:.2f} GB)"
        )
    setup_gpu_environment()
    log_gpu_info()
else:
    logger.warning(f"No GPU available: {gpu_info.get('error', 'Unknown reason')}")
    logger.warning("Training will proceed on CPU, which will be significantly slower")

DIR = get_name()
os.makedirs(DIR + "/data", exist_ok=True)

os.system(f"cp config.yaml {DIR}")

logger.info(f"Working directory: {DIR}")
logger.info("Loading spot price datasets")

lsd = LoadSpotDataset(f"{DIR}/config.yaml")

prices_df, instance_info_df = lsd.load_data()

train_df, val_df, test_df = lsd.get_training_validation_test_split(prices_df)

# Save dataframes to model-specific directory
prices_df.to_pickle(f"{DIR}/data/prices_df.pkl")
instance_info_df.to_pickle(f"{DIR}/data/instance_info_df.pkl")

train_df.to_pickle(f"{DIR}/data/train_df.pkl")
val_df.to_pickle(f"{DIR}/data/val_df.pkl")
test_df.to_pickle(f"{DIR}/data/test_df.pkl")

logger.info(f"Data saved to {DIR}")
logger.info(f"Pipeline started on {datetime.now()}")

# Get start and end dates for train_df
train_start_date = train_df["price_timestamp"].min()
train_end_date = train_df["price_timestamp"].max()
train_days = (train_end_date - train_start_date).days

# Get start and end dates for val_df
val_start_date = val_df["price_timestamp"].min()
val_end_date = val_df["price_timestamp"].max()
val_days = (val_end_date - val_start_date).days

# Get start and end dates for test_df
test_start_date = test_df["price_timestamp"].min()
test_end_date = test_df["price_timestamp"].max()
test_days = (test_end_date - test_start_date).days

logger.info(f"Train data: {train_start_date} to {train_end_date}, {train_days} days")
logger.info(f"Validation data: {val_start_date} to {val_end_date}, {val_days} days")
logger.info(f"Test data: {test_start_date} to {test_end_date}, {test_days} days")

train_dataset = SpotDataset(train_df, instance_info_df, DIR, training=True)
val_dataset = SpotDataset(val_df, instance_info_df, DIR, training=True)

# Pass train_dataset to get_model to automatically derive feature sizes
model = get_model(DIR, train_dataset)

logger.info(f"Training started at: {datetime.now()}")

modelTraining = Training(model, DIR)
modelTraining.train_model(train_dataset, val_dataset)

logger.info(f"Training completed at: {datetime.now()}")

model.save()

with open(f"{DIR}/training/training_history.json", "r") as f:
    history = json.load(f)

ResultPlotter(DIR).plot_training_history(history)

# Create test dataset before model initialization to derive feature sizes
test_dataset = SpotDataset(test_df, instance_info_df, DIR)

# Pass test_dataset to get_model to automatically derive feature sizes
logger.info("Loading model for evaluation")
model = get_model(DIR, test_dataset)
model.load()

logger.info("Starting model evaluation")
evaluator = Evaluate(model, test_dataset, DIR)
metrics = evaluator.evaluate_all()
evaluator.save_metrics()
logger.info("Evaluation completed successfully")
