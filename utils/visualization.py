import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional
from datetime import timedelta


def plot_forecast(
    model,
    input_df: pd.DataFrame,
    instance_id: Union[int, str],
    n_steps: int = 7,
    plot_history: bool = True,
    save_path: Optional[str] = None,
    date_column: str = "price_timestamp",
    value_column: str = "spot_price",
    id_column: str = "id_instance",
):
    """
    Generate and plot predictions for a specific instance.

    Args:
        model: The forecasting model (must have a forecast method)
        input_df: DataFrame containing historical data
        instance_id: ID of the instance to forecast
        n_steps: Number of steps to predict
        plot_history: Whether to plot all historical data or just recent points
        save_path: Path to save the plot image
        date_column: Name of the date/timestamp column
        value_column: Name of the value column to predict
        id_column: Name of the instance ID column

    Returns:
        pd.DataFrame: Combined dataframe with historical and predicted values
    """
    sequence_length = model.config.get("sequence_length", 30)

    instance_data = input_df[input_df[id_column] == instance_id].copy()
    instance_data = instance_data.sort_values(date_column)

    if len(instance_data) < sequence_length:
        raise ValueError(
            f"Not enough data for instance {instance_id}. Need at least {sequence_length} points."
        )

    input_seq = instance_data[value_column].values[-sequence_length:]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32)

    # Generate predictions
    predictions = model.forecast(input_tensor, n_steps, instance_id)[0]

    # Convert predictions to numpy if they're tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    last_date = instance_data[date_column].max()

    # Create dates for predictions (assumes daily steps, adjust as needed)
    pred_dates = [(last_date + timedelta(days=i + 1)) for i in range(n_steps)]

    # Create prediction dataframe
    pred_df = pd.DataFrame(
        {
            date_column: pred_dates,
            value_column: predictions,
            id_column: instance_id,
            "type": "prediction",
        }
    )

    # Add type to historical data
    instance_data["type"] = "historical"
    combined_df = pd.concat([instance_data, pred_df])

    # Create visualization
    plt.figure(figsize=(12, 6))

    if plot_history:
        # Plot all historical data
        plt.plot(
            instance_data[date_column],
            instance_data[value_column],
            "b-",
            label="Historical",
        )
    else:
        # Plot only recent history
        recent = instance_data.iloc[-sequence_length:]
        plt.plot(
            recent[date_column],
            recent[value_column],
            "b-",
            label="Historical",
        )

    # Plot predictions
    plt.plot(pred_df[date_column], pred_df[value_column], "r-", label="Prediction")

    # Add a vertical line at the prediction boundary
    plt.axvline(x=last_date, color="k", linestyle="--")

    # Add labels and title
    plt.title(f"Spot Price Prediction for Instance {instance_id}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Save if requested
    if save_path:
        plt.savefig(save_path)

    plt.show()
    return combined_df
