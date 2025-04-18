import torch
import numpy as np


def calculate_significant_trend_accuracy(
    predictions, targets, significance_threshold=0.02
):
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


def calculate_spot_price_savings(predictions, targets, decision_window=10):
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


def calculate_perfect_information_savings(targets, decision_window=10):
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

    # Calculate percentage savings - multiply by 100 to match calculate_spot_price_savings
    savings_pct = (
        100 * (immediate_costs.sum() - perfect_costs.sum()) / immediate_costs.sum()
    )

    return float(savings_pct)
