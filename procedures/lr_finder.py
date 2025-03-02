from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import SpotDataset
from model import Model
from loss import MultiStepForecastLoss
from utils import get_device


class LRFinder:
    """
    Learning rate finder implementation that helps determine optimal learning rate.

    Uses an exponential increase in learning rate while monitoring loss behavior
    to identify the optimal range where loss decreases most rapidly.
    """

    def __init__(
        self,
        model: Model,
        init_value: float = 1e-8,
        final_value: float = 1.0,
        beta: float = 0.98,
        output_dir: str = "output",
    ):
        """Initialize the learning rate finder with model and parameters."""
        self.model = model
        self.device = get_device()
        self.model.to(self.device)

        # Learning rate search parameters
        self.init_value = init_value
        self.final_value = final_value
        self.beta = beta  # Smoothing factor for loss

        # Output directory for saving results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Storage for results
        self.log_lrs: List[float] = []
        self.losses: List[float] = []

    def find(
        self, dataset: SpotDataset, batch_size: int = 32, weight_decay: float = 1e-4
    ) -> Tuple[float, Dict]:
        """
        Run the learning rate finder process.

        Args:
            dataset: Dataset to use for learning rate search
            batch_size: Batch size for training
            weight_decay: Weight decay factor for optimizer

        Returns:
            Tuple of optimal learning rate and results dictionary
        """
        # Create data loader
        train_loader = dataset.get_data_loader(batch_size=batch_size, shuffle=True)

        # Ensure model has a normalizer
        if self.model.normalizer is None and hasattr(dataset, "get_sequences"):
            print("Model has no normalizer. Creating one from dataset...")
            from procedures.train import Training

            trainer = Training(self.model)
            trainer.prepare_normalizer(dataset)

        # Set up model for training
        self.model.train()

        # Setup loss and optimizer
        criterion = MultiStepForecastLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.init_value,
            weight_decay=weight_decay,
        )

        # Reset state
        self.log_lrs = []
        self.losses = []

        # Calculate multiplication factor for LR increase
        num_batches = len(train_loader)
        mult = (self.final_value / self.init_value) ** (1 / num_batches)
        lr = self.init_value
        best_loss = float("inf")
        avg_loss = 0

        print(
            f"Starting LR search from {self.init_value:.2e} to {self.final_value:.2e}"
        )

        try:
            with tqdm(train_loader, desc="Finding LR", leave=True) as pbar:
                for batch_idx, (data, target) in enumerate(pbar):
                    # Move data to device
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    # Get instance IDs if available - needed for normalization
                    instance_ids = None
                    if hasattr(train_loader.dataset, "get_instance_id"):
                        batch_indices = list(
                            range(
                                batch_idx * batch_size,
                                min(
                                    (batch_idx + 1) * batch_size,
                                    len(train_loader.dataset),
                                ),
                            )
                        )
                        instance_ids = [
                            train_loader.dataset.get_instance_id(idx)
                            for idx in batch_indices
                        ]

                    # Forward pass
                    output = self.model(data)
                    loss, _ = criterion(output, target)

                    # Backward pass
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    # Apply exponential smoothing to the loss
                    current_loss = loss.item()
                    smoothed_loss = (
                        current_loss
                        if batch_idx == 0
                        else self.beta * self.losses[-1]
                        + (1 - self.beta) * current_loss
                    )

                    # Correct bias in the smoothed loss
                    bias_correction = 1 - self.beta ** (batch_idx + 1)
                    smoothed_loss = smoothed_loss / bias_correction

                    # Record results
                    self.log_lrs.append(np.log10(lr))
                    self.losses.append(smoothed_loss)

                    # Stop if loss explodes - prevents wasting time
                    if batch_idx > 0 and smoothed_loss > 4 * best_loss:
                        print("\nLoss exploding, stopping early...")
                        break

                    if smoothed_loss < best_loss:
                        best_loss = smoothed_loss

                    # Update progress bar
                    pbar.set_postfix(
                        {"lr": f"{lr:.2e}", "loss": f"{smoothed_loss:.6f}"}
                    )

                    # Update learning rate for next batch
                    lr *= mult
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

        except KeyboardInterrupt:
            print("\nLearning rate search interrupted by user")
        except Exception as e:
            print(f"\nError during learning rate search: {str(e)}")
            raise

        # Calculate the optimal learning rate
        optimal_lr = self._analyze_results()

        # Save and show plot
        self._plot_results(optimal_lr)

        return optimal_lr, {"log_lrs": self.log_lrs, "losses": self.losses}

    def _analyze_results(self) -> float:
        """
        Analyze LR finder results to determine optimal learning rate.

        The optimal learning rate is found where the loss reduction is steepest,
        which corresponds to the minimum of the gradient of smoothed losses.

        Returns:
            Optimal learning rate value
        """
        if len(self.losses) < 10:  # Need enough points for reliable gradient
            print("Warning: Not enough data points to reliably determine optimal LR")
            return self.init_value

        # Calculate gradients
        smoothed_losses = np.array(self.losses)
        gradients = np.gradient(smoothed_losses)

        # Find the point of steepest descent (most negative gradient)
        min_grad_idx = np.argmin(gradients)

        # The optimal LR is typically a bit lower than the point of steepest descent
        # to ensure stability, so we divide by 10
        optimal_idx = min(min_grad_idx, len(self.log_lrs) - 1)
        optimal_lr = 10 ** self.log_lrs[optimal_idx] / 10

        print(f"\nOptimal learning rate: {optimal_lr:.2e}")
        return optimal_lr

    def _plot_results(self, optimal_lr: float) -> None:
        """
        Plot and save the LR finder results.

        Args:
            optimal_lr: The determined optimal learning rate to mark on the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.log_lrs, self.losses)
        plt.xlabel("Log Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder Results")
        plt.grid(True, alpha=0.3)

        # Mark the optimal learning rate
        log_optimal_lr = np.log10(optimal_lr)
        if min(self.log_lrs) <= log_optimal_lr <= max(self.log_lrs):
            plt.axvline(
                x=log_optimal_lr,
                color="r",
                linestyle="--",
                label=f"Optimal LR: {optimal_lr:.2e}",
            )
            plt.legend()

        # Save plot
        plt.savefig(os.path.join(self.output_dir, "lr_finder_results.png"))
        plt.close()


def find_lr(
    model: Model,
    dataset: SpotDataset,
    init_value: float = 1e-8,
    final_value: float = 1.0,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    output_dir: str = "output",
) -> Tuple[float, Dict]:
    """
    Find optimal learning rate through exponential increase.

    This is a convenience wrapper around the LRFinder class.

    Args:
        model: The model to use for finding learning rate
        dataset: Dataset to use for the search
        init_value: Initial learning rate
        final_value: Final learning rate
        batch_size: Batch size for training
        weight_decay: Weight decay factor for optimizer
        output_dir: Directory to save results

    Returns:
        Tuple of (optimal_learning_rate, results_dict)
    """
    finder = LRFinder(
        model=model,
        init_value=init_value,
        final_value=final_value,
        output_dir=output_dir,
    )

    return finder.find(
        dataset=dataset, batch_size=batch_size, weight_decay=weight_decay
    )
