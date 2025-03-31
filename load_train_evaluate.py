# %%
from datetime import datetime
import os
import pandas as pd

from model import get_model
from procedures import Training, Evaluate
from dataset import SpotDataset, LoadSpotDataset
from utils import get_name

# %% [markdown]
# # Set up model directory structure

# %%
DIR = get_name()
os.makedirs(DIR + "/data", exist_ok=True)

os.system(f"cp config.yaml {DIR}")

# %%
lsd = LoadSpotDataset(f"{DIR}/config.yaml")

prices_df, instance_info_df = lsd.load_data()

# %%
train_df, val_df, test_df = lsd.get_training_validation_test_split(
    prices_df, train_ratio=0.7, val_ratio=0.15
)

# %%
# Save dataframes to model-specific directory
prices_df.to_pickle(f"{DIR}/data/prices_df.pkl")
instance_info_df.to_pickle(f"{DIR}/data/instance_info_df.pkl")

train_df.to_pickle(f"{DIR}/data/train_df.pkl")
val_df.to_pickle(f"{DIR}/data/val_df.pkl")
test_df.to_pickle(f"{DIR}/data/test_df.pkl")

print(f"Data saved to {DIR}")
print(f"Created on {datetime.now()}")

# %%
def display_df_stats(df, name):
    """Helper function to display DataFrame statistics"""
    print(f"\n=== {name} Statistics ===")
    print("\nShape:", df.shape)
    print("\nInfo:")
    df.info()
    print("\nSample Data:")
    print(df.head())
    if "price_timestamp" in df.columns:
        start_date = df["price_timestamp"].min()
        end_date = df["price_timestamp"].max()
        days = (end_date - start_date).days
        print(f"\nDate Range: {start_date} to {end_date} ({days} days)")


# %%
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

print(
    f"Train DataFrame: Start Date = {train_start_date}, End Date = {train_end_date}, Number of Days = {train_days}"
)
print(
    f"Validation DataFrame: Start Date = {val_start_date}, End Date = {val_end_date}, Number of Days = {val_days}"
)
print(
    f"Test DataFrame: Start Date = {test_start_date}, End Date = {test_end_date}, Number of Days = {test_days}"
)

# %%
display_df_stats(prices_df, "Prices DataFrame")
display_df_stats(instance_info_df, "Instance Info DataFrame")
display_df_stats(train_df, "Training Set")
display_df_stats(val_df, "Validation Set")
display_df_stats(test_df, "Test Set")

train_dataset = SpotDataset(train_df, DIR)
val_dataset = SpotDataset(val_df, DIR)

# %% [markdown]
# # Model Training

# %%
model = get_model(DIR)

print(f"Training started at: {datetime.now()}")

modelTraining = Training(model, DIR)
modelTraining.train_model(train_dataset, val_dataset)
print(f"Training endend at: {datetime.now()}")
# %%
model.save()

evaluator = Evaluate(model, DIR)

# %%
metrics = evaluator.evaluate_all(test_df)
print(f"Evaluation endend at: {datetime.now()}")


# %%
def dump_metrics_to_csv(
    evaluate_instance: Evaluate, instance_info_df: pd.DataFrame, output_dir
):
    """Dump segmented metrics to a CSV file with instance properties"""
    output_file = os.path.join(output_dir, "evaluation_metrics.csv")

    # Get metrics from Evaluate class
    metric_columns = evaluate_instance.metrics

    # Flatten the metrics data
    rows = []
    for instance_id, metrics_list in evaluate_instance.segmented_metrics.items():
        # Get instance properties
        instance_props = instance_info_df.loc[
            instance_id,
            ["region", "av_zone", "instance_type", "generation", "modifiers", "size"],
        ].to_dict()  # type: ignore

        for metric in metrics_list:
            row = {
                "instance_id": instance_id,
                **instance_props,  # Unpack instance properties
                **{
                    col: metric[col] for col in metric_columns
                },  # Unpack metrics using the defined columns
            }
            rows.append(row)

    # Convert to DataFrame and save to CSV
    metrics_df = pd.DataFrame(rows)
    column_order = [
        "instance_id",
        "region",
        "av_zone",
        "instance_type",
        "generation",
        "modifiers",
        "size",
    ] + metric_columns

    metrics_df = metrics_df[column_order]
    metrics_df.to_csv(output_file, index=False)


dump_metrics_to_csv(evaluator, instance_info_df, DIR + "/evaluation")
