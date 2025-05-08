import torch
import numpy as np
import math


def mse_loss(y_pred, y_true):
    """
    Calculate the Mean Squared Error loss between predictions and targets.
    Preserves computational graph for gradient backpropagation.

    Args:
        y_pred: Model predictions tensor
        y_true: Ground truth values tensor

    Returns:
        MSE loss tensor for backpropagation
    """
    return torch.mean((y_pred - y_true) ** 2)


def mean_squared_error(y_pred, y_true):
    """
    Calculate the Mean Squared Error between predictions and targets.

    Args:
        y_pred: Model predictions tensor
        y_true: Ground truth values tensor

    Returns:
        MSE value as a scalar
    """
    return torch.mean((y_pred - y_true) ** 2).item()


def root_mean_squared_error(y_pred, y_true):
    """
    Calculate the Root Mean Squared Error between predictions and targets.

    Args:
        y_pred: Model predictions tensor
        y_true: Ground truth values tensor

    Returns:
        RMSE value as a scalar
    """
    return math.sqrt(mean_squared_error(y_pred, y_true))


def mean_absolute_percentage_error(y_pred, y_true):
    """
    Calculate the Mean Absolute Percentage Error between predictions and targets.

    Args:
        y_pred: Model predictions tensor
        y_true: Ground truth values tensor

    Returns:
        MAPE value as a percentage (0-100 scale)
    """
    # For numerical stability
    epsilon = 1e-8

    # Clip values to avoid extreme outliers
    y_true_safe = torch.clamp(torch.abs(y_true), min=epsilon)
    abs_percentage_error = torch.abs((y_true - y_pred) / y_true_safe)

    # Clip extremely large percentage errors for stability
    abs_percentage_error = torch.clamp(abs_percentage_error, max=10.0)

    # Return as percentage
    return torch.mean(abs_percentage_error).item() * 100


def differentiable_trend_loss(
    pred_diff, true_diff, significance_threshold=0.02, smoothing_factor=10.0
):
    """
    Computes a differentiable loss for trend prediction that mimics significant_trend_accuracy
    but maintains gradient flow for training.

    Args:
        pred_diff: Predicted price differences (batch_size, timesteps-1)
        true_diff: Actual price differences (batch_size, timesteps-1)
        significance_threshold: Threshold for significant changes as fraction of mean price
        smoothing_factor: Controls the sharpness of the approximation

    Returns:
        Differentiable trend loss tensor (scalar)
    """
    # Stability constant
    epsilon = 1e-8

    # Compute significance weights with a smooth transition at the threshold
    # Higher values have weight closer to 1, lower values closer to 0
    significance_magnitude = torch.abs(true_diff)
    mean_price = torch.mean(torch.abs(true_diff), dim=1, keepdim=True)
    threshold = significance_threshold * mean_price

    # Sigmoid gives a smooth transition from 0 to 1 around the threshold
    significance_weight = torch.sigmoid(
        (significance_magnitude - threshold) * smoothing_factor
    )

    # Get the signs of the differences (direction of change)
    # Using tanh for a differentiable approximation of the sign function
    pred_sign = torch.tanh(smoothing_factor * pred_diff)
    true_sign = torch.tanh(smoothing_factor * true_diff)

    # Compute direction matching score: 1 when signs match, 0 when opposite
    # (true_sign * pred_sign + 1) / 2 maps from [-1,1] to [0,1]
    direction_match = (true_sign * pred_sign + 1) / 2

    # Apply significance weights - only significant changes affect the loss
    weighted_direction_match = direction_match * significance_weight

    # Normalize by the sum of significance weights to get an accuracy-like score
    significance_sum = torch.sum(significance_weight) + epsilon
    trend_accuracy = torch.sum(weighted_direction_match) / significance_sum

    # Loss is 1 - accuracy (we want to maximize accuracy -> minimize loss)
    trend_loss = 1.0 - trend_accuracy

    return trend_loss


def calculate_savings_efficiency(cost_savings, perfect_savings):
    """
    Calculate how efficient our cost savings are compared to perfect information.

    Args:
        cost_savings: Cost savings percentage achieved by model
        perfect_savings: Maximum theoretical cost savings with perfect information

    Returns:
        Savings efficiency as a percentage (0-100 scale)
    """
    if perfect_savings > 0:
        return (cost_savings / perfect_savings) * 100
    else:
        # If perfect savings is 0, there's no room for improvement
        return 100.0 if cost_savings == 0 else 0.0


def significant_trend_accuracy(predictions, targets, significance_threshold=0.02):
    """
    Calculate accuracy on predicting the direction of significant price movements.
    Focuses only on changes that exceed the significance threshold.

    Args:
        predictions: Model predictions tensor
        targets: Ground truth values tensor
        significance_threshold: Minimum percentage change to consider significant

    Returns:
        Significant trend accuracy as a percentage (0-100 scale)
    """
    # Calculate price changes
    epsilon = 1e-6
    target_abs = torch.clamp(torch.abs(targets[:, :-1]), min=epsilon)

    # Calculate percentage changes
    target_change = (targets[:, 1:] - targets[:, :-1]) / target_abs
    pred_change = (predictions[:, 1:] - predictions[:, :-1]) / target_abs

    # Identify significant moves (exceeding threshold)
    significant_mask = torch.abs(target_change) >= significance_threshold

    # If no significant moves were found, return 100 (perfect score)
    if not torch.any(significant_mask):
        return 100.0

    # Direction match (1 if correct, 0 if wrong)
    direction_match = (torch.sign(pred_change) == torch.sign(target_change)).float()

    # Calculate accuracy only for significant moves
    sig_accuracy = direction_match[significant_mask].mean().item() * 100

    return sig_accuracy


def spot_price_savings(predictions, targets, decision_window=10):
    """
    Simulate EC2 spot price task scheduling decisions and calculate cost savings.

    This metric models the decision process of "execute now vs. wait" for tasks
    that can be delayed up to a specified window, and calculates the resulting
    actual cost savings compared to immediate execution.

    Args:
        predictions: Model predictions tensor
        targets: Ground truth values tensor
        decision_window: Maximum number of time periods to delay task execution

    Returns:
        Percentage cost savings from using model predictions
    """
    # Ensure the sequence is long enough for meaningful analysis
    seq_length = targets.shape[1]
    if seq_length <= decision_window:
        decision_window = seq_length - 1

    # Will store actual costs for each sequence
    immediate_costs = []
    optimized_costs = []

    # For each sequence in the batch
    for i in range(targets.shape[0]):
        # Process multiple decision points in each sequence
        for start_idx in range(seq_length - decision_window):
            # Current window of predictions and actual prices
            pred_window = predictions[i, start_idx : start_idx + decision_window]
            target_window = targets[i, start_idx : start_idx + decision_window]

            # Current price (what we'd pay if executing immediately)
            current_price = target_window[0].item()
            immediate_costs.append(current_price)

            # Find when the model predicts the minimum price will occur
            min_idx = torch.argmin(pred_window).item()
            # Use the actual price at that time (whether it's now or in the future)
            optimized_costs.append(target_window[min_idx].item())

    # Calculate average percentage savings
    if len(immediate_costs) == 0 or sum(immediate_costs) == 0:
        return 0.0

    # Convert to numpy for calculation
    immediate_costs = np.array(immediate_costs)
    optimized_costs = np.array(optimized_costs)

    # Calculate percentage savings
    savings_pct = (
        100 * (immediate_costs.sum() - optimized_costs.sum()) / immediate_costs.sum()
    )

    return float(savings_pct)


def perfect_information_savings(targets, decision_window=10):
    """
    Calculate the maximum theoretical cost savings with perfect price prediction.

    This metric represents the upper bound of cost savings possible if we had
    perfect knowledge of future prices and always executed at the minimum price
    within each decision window.

    Args:
        targets: Ground truth values tensor
        decision_window: Maximum number of time periods to delay task execution

    Returns:
        Percentage of maximum possible cost savings with perfect information
    """
    # Ensure the sequence is long enough for meaningful analysis
    seq_length = targets.shape[1]
    if seq_length <= decision_window:
        decision_window = seq_length - 1

    # Will store actual costs for each sequence
    immediate_costs = []
    perfect_costs = []

    # For each sequence in the batch
    for i in range(targets.shape[0]):
        # Process multiple decision points in each sequence
        for start_idx in range(seq_length - decision_window):
            # Current window of actual prices
            target_window = targets[i, start_idx : start_idx + decision_window]

            # Current price (what we'd pay if executing immediately)
            current_price = target_window[0].item()
            immediate_costs.append(current_price)

            # With perfect information, we would always choose the minimum price in the window
            min_price = torch.min(target_window).item()
            perfect_costs.append(min_price)

    # Calculate average percentage savings
    if len(immediate_costs) == 0 or sum(immediate_costs) == 0:
        return 0.0

    # Convert to numpy for calculation
    immediate_costs = np.array(immediate_costs)
    perfect_costs = np.array(perfect_costs)

    # Calculate percentage savings - multiply by 100 to match spot_price_savings
    savings_pct = (
        100 * (immediate_costs.sum() - perfect_costs.sum()) / immediate_costs.sum()
    )

    return float(savings_pct)
