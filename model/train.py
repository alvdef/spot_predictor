from typing import Dict, List
from datetime import datetime
import torch
from tqdm import tqdm
import os

from utils.model_utils import CombinedLoss
from .checkpoint_tracker import CheckpointTracker
from .metrics_tracker import MetricsTracker
from utils import get_device, load_config


class Training:
    REQUIRED_CONFIG_FIELDS = [
        "mse_weight",
        "learning_rate",
        "weight_decay",
        "adam_betas",
        "max_learning_rate",
        "epochs",
        "steps_per_epoch",
    ]

    def __init__(self, model: torch.nn.Module, config_path: str = "config.yaml"):
        self.device = get_device()
        self.model = model

        # Setup output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize trackers
        self.metrics = MetricsTracker(self.output_dir)
        self.checkpoints = CheckpointTracker()

        # Try load previous checkpoint
        prev_config = None
        try:
            prev_config, prev_best_loss = self.checkpoints.load(self.model)
            if prev_config:
                self.metrics.best_loss = prev_best_loss
        except Exception as e:
            print(f"Warning: Could not load checkpoint - {str(e)}")
            print("Continuing with new model initialization.")

        # Load or use previous config
        self.config = (
            prev_config
            if prev_config
            else load_config(
                config_path, "training_hyperparams", self.REQUIRED_CONFIG_FIELDS
            )
        )

        # Initialize criterion and optimizer
        self.criterion = CombinedLoss(mse_weight=self.config["mse_weight"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            betas=self.config["adam_betas"],
        )

        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["max_learning_rate"],
            epochs=self.config["epochs"],
            steps_per_epoch=self.config["steps_per_epoch"],
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
        )

    @property
    def history(self) -> Dict[str, List[float]]:
        return self.metrics.history

    def train_model(self, train_loader, val_loader=None):
        print("\n" + "=" * 50)
        print(f"Training for {self.config['epochs']} epochs")
        print("=" * 50 + "\n")

        try:
            for epoch in range(self.config["epochs"]):
                start_epoch = datetime.now()

                # Training phase
                current_train_loss, current_lr, grad_norm = self.train_epoch(
                    train_loader
                )

                # Validation phase
                current_val_loss = None
                if val_loader is not None:
                    current_val_loss = self.validate_model(val_loader)

                # Update tracking
                epoch_duration = (datetime.now() - start_epoch).total_seconds()
                self.metrics.update_history(
                    current_train_loss,
                    current_lr,
                    grad_norm.item(),
                    epoch_duration,
                    current_val_loss,
                )

                # Save model state if improved
                model_state = {
                    "model_state_dict": self.model.state_dict(),
                    "loss": current_val_loss or current_train_loss,
                    "config": self.config,
                }
                self.checkpoints.save_if_best(model_state)

                # Print progress
                val_str = f"{current_val_loss:.4f}" if current_val_loss else "N/A"
                print(
                    f"Epoch {epoch+1}/{self.config['epochs']}\n"
                    f"T: {current_train_loss:.4f}  |  V: {val_str}  |  "
                    f"LR: {current_lr:.1e}  |  GN: {grad_norm:.1f}  |  "
                    f"{epoch_duration:.1f}s"
                )

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            print("Saving current model state...")
            model_state = {
                "model_state_dict": self.model.state_dict(),
                "loss": self.metrics.current_val_loss or self.metrics.current_loss,
                "config": self.config,
            }
            self.checkpoints.save(model_state, is_best=False)
            self.metrics.save_to_files()

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            self.metrics.save_to_files()
            raise

        self.metrics.save_to_files()

    def train_epoch(self, train_loader):
        """Train model for one epoch with improved efficiency"""
        self.model.train()
        epoch_loss = torch.tensor(0.0, device=self.device)

        scaler = torch.amp.grad_scaler.GradScaler(enabled=torch.cuda.is_available())

        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for data, target in pbar:
                # Move data to device efficiently
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                output = self.model(data)
                loss = self.criterion(output, target)

                # Efficient backward pass
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                # Optimizer step with mixed precision
                scaler.step(self.optimizer)
                scaler.update()

                # Learning rate scheduling
                self.scheduler.step()

                # Update metrics efficiently
                epoch_loss += loss
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        train_loss = (epoch_loss / len(train_loader)).item()
        current_lr = self.optimizer.param_groups[0]["lr"]

        return train_loss, current_lr, grad_norm

    def validate_model(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)
