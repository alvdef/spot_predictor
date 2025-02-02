import numpy as np
import torch

from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


class CombinedLoss(torch.nn.Module):
    def __init__(self, mse_weight=0.7):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()

    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + (
            1 - self.mse_weight
        ) * self.mae(pred, target)


def find_lr(model, train_loader, config):
    device = get_device()

    """Find optimal learning rate through exponential increase"""
    init_value = config.get("init_learning_rate", 1e-8)
    final_value = config.get("final_learning_rate", 1)
    beta = 0.98  # Smoothing factor

    # Setup model and optimizer with initial learning rate
    model = model.to(device)
    model.train()
    criterion = CombinedLoss(mse_weight=config.get("mse_weight", 0.7))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=init_value,
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Initialize tracking variables
    num_batches = len(train_loader)
    mult = (final_value / init_value) ** (1 / num_batches)
    lr = init_value
    best_loss = float("inf")
    losses = []
    log_lrs = []
    avg_loss = 0

    try:
        with tqdm(train_loader, desc="Finding LR", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                # Move data to device
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # Forward pass
                output = model(data)
                loss = criterion(output, target)

                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Update weights and learning rate
                optimizer.step()

                # Smooth the loss
                current_loss = loss.item()
                smoothed_loss = (
                    current_loss
                    if batch_idx == 0
                    else beta * losses[-1] + (1 - beta) * current_loss
                )

                # Early stopping on loss explosion
                if batch_idx > 0 and smoothed_loss > 4 * best_loss:
                    print("\nLoss exploding, stopping early...")
                    break

                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss

                # Log results
                losses.append(smoothed_loss)
                log_lrs.append(np.log10(lr))

                # Update progress bar
                pbar.set_postfix({"lr": f"{lr:.2e}", "loss": f"{smoothed_loss:.6f}"})

                # Update learning rate for next batch
                lr *= mult
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

        # Find the point of steepest loss decline
        smoothed_losses = np.array(losses)
        min_grad_idx = np.gradient(smoothed_losses).argmin()
        optimal_lr = 10 ** log_lrs[min_grad_idx]

        print(f"\nOptimal learning rate: {optimal_lr:.2e}")
        return log_lrs, losses

    except KeyboardInterrupt:
        print("\nLearning rate search interrupted by user")
        return log_lrs, losses
    except Exception as e:
        print(f"\nError during learning rate search: {str(e)}")
        raise
