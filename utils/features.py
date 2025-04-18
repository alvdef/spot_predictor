from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
import torch


def extract_time_features(
    timestamps: pd.Series, feature_names: List[str]
) -> np.ndarray:
    """
    Extract cyclical time features from timestamps using sine/cosine encoding.

    This function converts temporal features (hour, day, month, etc.) into
    cyclical representations using sine and cosine transformations,
    which preserves the circular nature of time periods.

    Args:
        timestamps: Series of timestamps to extract features from
        feature_names: List of features to extract, supported options are:
                      ['hour', 'dayofweek', 'dayofmonth', 'month', 'dayofyear']

    Returns:
        Array of shape (len(timestamps), num_features*2) containing sine/cosine values
    """
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)

    feature_columns = []

    # Define periods for each time feature
    time_features = {
        "hour": (timestamps.dt.hour.values, 24),
        "dayofweek": (timestamps.dt.dayofweek.values, 7),
        "dayofmonth": (np.array(timestamps.dt.day.values) - 1, 31),
        "month": (np.array(timestamps.dt.month.values) - 1, 12),
        "dayofyear": (np.array(timestamps.dt.dayofyear.values) - 1, 365),
    }

    # Process each requested feature
    for feature, (values, period) in time_features.items():
        if feature in feature_names:
            # Convert to radians with appropriate period
            values_normalized = values * (2 * np.pi / period)

            # Encode using both sine and cosine for continuity
            cos_values = np.cos(values_normalized)
            sin_values = np.sin(values_normalized)

            # Add to output columns
            feature_columns.append(cos_values)
            feature_columns.append(sin_values)

    if not feature_columns:
        raise ValueError(f"No valid time features found in {feature_names}")

    return np.column_stack(feature_columns).astype(np.float32)


def predict_future_time_features(
    time_features: torch.Tensor,
    feature_names: List[str],
    time_delta_hours: float,
    num_steps: int,
) -> torch.Tensor:
    """
    Predict future time features by advancing encoded cyclical time features.

    This function directly advances the sine/cosine encoded time features by
    calculating the appropriate angular increment based on the time delta.
    Works entirely with tensors for better performance, especially on GPU.

    Args:
        time_features: Tensor of shape (batch_size, seq_len, num_features*2)
                      containing encoded time features (cosine,sine pairs)
        feature_names: List of time features in order: ['hour', 'dayofweek', etc.]
        time_delta_hours: Time difference in hours between consecutive steps
        num_steps: Number of future steps to predict

    Returns:
        Tensor of shape (batch_size, num_steps, num_features*2) with predicted features
    """
    # Get the dimensions and device
    batch_size, _, feature_dim = time_features.shape
    device = time_features.device

    # Extract the last time features for each sequence
    last_features = time_features[:, -1, :]

    # Define angular velocities (radians per hour) for each feature
    angular_velocities = {
        "hour": 2 * torch.pi / 24,  # 24 hours per cycle
        "dayofweek": 2 * torch.pi / (7 * 24),  # 7 days per cycle
        "dayofmonth": 2 * torch.pi / (30 * 24),  # ~30 days per cycle
        "month": 2 * torch.pi / (365 * 24),  # ~365 days per cycle
        "dayofyear": 2 * torch.pi / (365 * 24),  # 365 days per cycle
    }

    # Create output tensor directly on the same device
    future_features = torch.zeros(
        (batch_size, num_steps, feature_dim), dtype=torch.float32, device=device
    )

    # For each feature pair in the input
    feature_idx = 0
    for feature in feature_names:
        if feature not in angular_velocities:
            continue

        # Get angular velocity for this feature
        velocity = angular_velocities[feature]

        # Process all batches at once using tensor operations
        for step in range(num_steps):
            # Time advancement for this step
            delta_angle = velocity * time_delta_hours * (step + 1)

            # Get current cos/sin values for all batches at once
            cos_val = last_features[:, feature_idx]
            sin_val = last_features[:, feature_idx + 1]

            # Calculate current angle using atan2
            current_angle = torch.atan2(sin_val, cos_val)

            # Advance angle and calculate new cos/sin in one operation
            new_angle = current_angle + delta_angle
            future_features[:, step, feature_idx] = torch.cos(new_angle)
            future_features[:, step, feature_idx + 1] = torch.sin(new_angle)

        # Move to next feature pair
        feature_idx += 2

    return future_features
