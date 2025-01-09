import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from tqdm import tqdm
from tqdm.auto import trange


class SpotBiLSTM(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        self.hidden_size = model_config["hidden_size"]
        self.n_layers = model_config["n_layers"]
        self.output_scale = model_config["output_scale"]

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Final dense layer
        self.dense = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # First LSTM layer
        x, _ = self.lstm1(x)

        # Second LSTM layer
        x, _ = self.lstm2(x)

        # Take only the last output
        x = x[:, -1, :]

        # Dense layer
        x = self.dense(x)

        # Scale output
        x = x * self.output_scale

        return x


def setup_training(model, config, device):
    """Initialize model, optimizer, criterion and scheduler for training"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["optimizer"]["momentum"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    return model, criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    num_batches = len(train_loader)

    # Use tqdm for batch progress
    pbar = tqdm(train_loader, leave=False)
    for data, target in pbar:
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

        # Update progress bar
        pbar.set_description(f"Training loss: {loss.item():.6f}")

    return epoch_loss / num_batches


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


def log_metrics(history, train_loss, val_loss=None, lr=None):
    """Log training metrics"""
    history["train_loss"].append(train_loss)
    if val_loss is not None:
        history["val_loss"].append(val_loss)
    if lr is not None:
        history["lr"].append(lr)
    return history


class TrainingLogger:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_messages = []
        
    def print_header(self):
        """Print global training information header"""
        print("\n" + "="*80)
        print(f"Training for {self.total_epochs} epochs")
        print("="*80 + "\n")
        
    def log_epoch(self, epoch, train_loss, val_loss=None, lr=None):
        """Log epoch results and store message"""
        elapsed = time.time() - self.start_time
        eta = (elapsed / (epoch + 1)) * (self.total_epochs - (epoch + 1)) if epoch > 0 else 0
        
        msg = f"Epoch [{epoch+1}/{self.total_epochs}] "
        msg += f"Train Loss: {train_loss:.6f} "
        if val_loss:
            msg += f"Val Loss: {val_loss:.6f} "
        if lr:
            msg += f"LR: {lr:.2e} "
        msg += f"│ Elapsed time: {timedelta(seconds=int(elapsed))} | Estimated remaining time: {timedelta(seconds=int(eta))}"
        
        self.epoch_messages.append(msg)
        print(msg)
        
    def _display_progress(self):
        """Clear screen and redisplay all messages"""
        # Clear previous output (ANSI escape sequence)
        print("\033[2J\033[H", end="")
        # Print header
        self.print_header()
        # Print all epoch messages
        for msg in self.epoch_messages:
            print(msg)


def train_model(model, train_loader, config, device, val_loader=None):
    """Train model with early stopping and learning rate scheduling"""
    # Setup
    model, criterion, optimizer, scheduler = setup_training(model, config, device)
    history = {"train_loss": [], "val_loss": [], "lr": []}
    
    # Initialize logger
    logger = TrainingLogger(config["epochs"])
    logger.print_header()

    # Early stopping setup
    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = 10

    try:
        for epoch in range(config["epochs"]):
            # Train one epoch
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = validate_model(model, val_loader, criterion, device)
                scheduler.step(val_loss)

                # Check early stopping
                stop_training, best_val_loss, patience_counter = check_early_stopping(
                    val_loss, best_val_loss, patience_counter, patience_limit
                )

                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), "best_model.pth")
                    print("New best model saved!")
                
                if stop_training:
                    logger.log_epoch(epoch, train_loss, val_loss, 
                                  optimizer.param_groups[0]["lr"])
                    print("\nEarly stopping triggered")
                    break

            # Log metrics
            current_lr = optimizer.param_groups[0]["lr"]
            history = log_metrics(history, train_loss, val_loss, current_lr)
            
            # Log epoch
            logger.log_epoch(epoch, train_loss, val_loss, current_lr)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

    # Print final summary
    elapsed = time.time() - logger.start_time
    print("\n" + "="*80)
    print(f"Training completed in {timedelta(seconds=int(elapsed))}")
    print("="*80 + "\n")

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


def find_lr(model, train_loader, device, init_value=1e-8, final_value=1, beta=0.98):
    """Perform learning rate range test to find optimal learning rate.

    Implementation of the learning rate range test proposed by Leslie Smith
    in the paper "Cyclical Learning Rates for Training Neural Networks"

    Args:
        model: Neural network model
        train_loader: DataLoader with training data
        device: Computing device (cpu/cuda/mps)
        init_value: Starting learning rate
        final_value: Maximum learning rate to test
        beta: Smoothing factor for loss tracking

    Returns:
        tuple: (log_learning_rates, smoothed_losses)
    """
    # Setup
    model, criterion, optimizer, _ = setup_training(
        model, {"learning_rate": init_value, "optimizer": {"momentum": 0.9}}, device
    )

    # Initialize tracking variables
    num_batches = len(train_loader)
    mult = (final_value / init_value) ** (1 / num_batches)  # LR multiplier
    lr = init_value
    losses, log_lrs = [], []
    avg_loss = min_loss = float("inf")
    optimal_lr = None

    # Use tqdm for progress tracking
    with tqdm(train_loader, desc="Finding LR", leave=False) as pbar:
        for batch_idx, (data, target) in enumerate(pbar, 1):
            # Skip empty batches
            if data.numel() == 0 or target.numel() == 0:
                continue

            # Training step
            optimizer.zero_grad()
            loss = criterion(model(data.to(device)), target.to(device))
            loss.backward()
            optimizer.step()

            # Update smoothed loss
            curr_loss = loss.item()
            avg_loss = beta * avg_loss + (1 - beta) * curr_loss
            smoothed_loss = avg_loss / (1 - beta**batch_idx)

            # Track metrics
            if smoothed_loss < min_loss:
                min_loss = smoothed_loss
                optimal_lr = lr

            losses.append(smoothed_loss)
            log_lrs.append(np.log10(lr))

            # Update progress bar
            pbar.set_postfix({"lr": f"{lr:.2e}", "loss": f"{smoothed_loss:.6f}"})

            # Check for loss explosion
            if batch_idx > 1 and smoothed_loss > 4 * min_loss:
                print("Loss explosion detected. Stopping early.")
                break

            # Update learning rate
            lr *= mult
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    print(f"Suggested learning rate: {optimal_lr:.2e}")
    return log_lrs, losses
