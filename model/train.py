import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import save_checkpoint, load_checkpoint, CombinedLoss


def setup_training(model, config, device):
    """Initialize model, optimizer, criterion and scheduler for training"""
    model = model.to(device)

    # Use a combination of MSE and MAE loss for better stability
    criterion = CombinedLoss(mse_weight=config.get("mse_weight", 0.7))

    # Use AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
        betas=(0.9, 0.999),
    )

    # Use OneCycleLR with proper total steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
    )

    return model, criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """Train model for one epoch with improved efficiency"""
    model.train()
    epoch_loss = torch.tensor(0.0, device=device)

    # Use automatic mixed precision for faster training
    scaler = torch.amp.grad_scaler.GradScaler()

    with tqdm(train_loader, desc="Training", leave=False) as pbar:
        for data, target in pbar:
            # Move data to device efficiently
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward pass with mixed precision
            output = model(data)
            loss = criterion(output, target)

            # Efficient backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with mixed precision
            scaler.step(optimizer)
            scaler.update()

            # Learning rate scheduling
            scheduler.step()

            # Update metrics efficiently
            epoch_loss += loss
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    train_loss = (epoch_loss / len(train_loader)).item()
    current_lr = optimizer.param_groups[0]["lr"]

    return train_loss, current_lr, grad_norm


def check_early_stopping(val_loss, best_val_loss, patience_counter, patience_limit):
    """Check if early stopping criteria is met"""
    stop_training = False
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            stop_training = True

    return stop_training, best_val_loss, patience_counter


def train_model(model, train_loader, config, device, val_loader=None):
    # Setup training components
    model, criterion, optimizer, scheduler = setup_training(model, config, device)

    # Load previous best model if it exists
    prev_config, best_loss = load_checkpoint(model)
    config = prev_config if prev_config else config

    # Initialize training state
    history = {"train_loss": [], "val_loss": [], "learning_rates": [], "grad_norms": []}

    print("\n" + "=" * 50)
    print(f"Training for {config['epochs']} epochs")
    print("=" * 50 + "\n")

    try:
        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0

            with tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False
            ) as pbar:
                train_loss, current_lr, grad_norm = train_epoch(
                    model, train_loader, criterion, optimizer, scheduler, device
                )
                model_state = {
                    "model_state_dict": model.state_dict(),
                    "loss": train_loss,
                    "config": config,
                }
            if val_loader is not None:
                val_loss = validate_model(model, val_loader, criterion, device)

                # Update learning rate for plateau scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

                model_state = {
                    "model_state_dict": model.state_dict(),
                    "loss": val_loss,
                    "config": config,
                }

            save_checkpoint(model_state, best_loss < model_state["loss"])
            best_loss = (
                best_loss if best_loss < model_state["loss"] else model_state["loss"]
            )

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rates"].append(current_lr)
            history["grad_norms"].append(grad_norm)

            # Print informative epoch summary
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"├─ Train Loss: {train_loss:.6f}")
            if val_loss is not None:
                print(f"├─ Val Loss: {val_loss:.6f}")
            print(f"├─ Learning Rate: {current_lr:.2e}")
            print(f"└─ Gradient Norm: {grad_norm:.2f}")

    except KeyboardInterrupt:
        model_state = {
            "model_state_dict": model.state_dict(),
            "loss": val_loss,
            "config": config,
        }
        print("\nTraining interrupted by user")
        print("Saving current model state...")
        save_checkpoint(model_state, False)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

    return history


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def model_forecast(model, series, window_size, batch_size, device):
    """Generate predictions using sliding windows"""
    model = model.to(device)
    model.eval()

    windows = []
    for i in range(len(series) - window_size + 1):
        window = series[i : i + window_size]
        windows.append(window)

    windows = torch.tensor(windows, dtype=torch.float32)
    dataset = DataLoader(windows, batch_size=batch_size)

    forecasts = []
    with torch.no_grad():
        for batch in dataset:
            batch = batch.to(device)
            forecast = model(batch)
            forecasts.append(forecast.cpu())

    return torch.cat(forecasts).numpy().squeeze()


def find_lr(model, train_loader, config, device):
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
