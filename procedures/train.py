from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import MultiStepForecastLoss
from utils import get_device, load_config, MetricsTracker, CheckpointTracker
from dataset import SpotDataset
from dataset.normalizer import Normalizer
from model import Model


class Training:
    REQUIRED_CONFIG_FIELDS = [
        "mse_weight",
        "learning_rate",
        "weight_decay",
        "max_learning_rate",
        "epochs",
        "patience",
        "pct_start",
        "div_factor",
    ]

    def __init__(
        self,
        model: Model,
        config_path: str = "config.yaml",
    ):
        """Initialize training process for the model."""
        self.device = get_device()
        self.model = model
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics = MetricsTracker(self.output_dir)
        self.checkpoints = CheckpointTracker()

        # Try to load previous checkpoint to continue training
        prev_config, prev_best_loss = self.checkpoints.load(self.model)
        if prev_config:
            self.metrics.best_loss = prev_best_loss
            self.config = prev_config
            print("Using previous configuration and model for training.")
        else:
            self.config = load_config(
                config_path, "training_hyperparams", self.REQUIRED_CONFIG_FIELDS
            )

        self._convert_numeric_params()

        self.metrics.early_stopping_patience = self.config["patience"]

        # Initialize loss function with configurable weights
        self.criterion = MultiStepForecastLoss(
            mse_weight=self.config.get("mse_weight", 0.4),
            mape_weight=self.config.get("mape_weight", 0.2),
            direction_weight=self.config.get("direction_weight", 0.4),
        )

        # AdamW optimizer tends to work better than Adam for time series models
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999),  # Default betas work well for most time series tasks
        )

    def prepare_normalizer(self, dataset: SpotDataset) -> None:
        """
        Create and fit normalizer using all available data for each instance.
        Uses the full sequence data to compute more stable normalization parameters.
        """
        print("Preparing normalizer for the model...")

        normalizer = Normalizer(device=self.device)

        all_data = dataset.get_sequences()
        instance_ids = all_data["instance_ids"]
        sequences = all_data["sequences"]  # Already on GPU as a single tensor

        # Calculate normalization parameters per instance using unique instance IDs
        unique_ids = list(set(instance_ids))
        for instance_id in unique_ids:
            # Get all sequences for this instance
            instance_mask = [
                i for i, iid in enumerate(instance_ids) if iid == instance_id
            ]
            instance_data = sequences[instance_mask]  # Shape: [n_sequences, seq_len, 1]

            # Remove feature dimension and combine all sequences
            instance_values = instance_data.squeeze(-1).reshape(-1)  # Flatten to 1D

            # Fit normalizer with all values from this instance
            normalizer.fit(instance_id, instance_values)

        stats = normalizer.get_params_summary()
        print(f"Normalizer prepared for {stats['count']} instances")

        # Attach normalizer to model
        self.model.attach_normalizer(normalizer)
        normalizer.save(os.path.join(self.output_dir, "normalizer.json"))

    @property
    def history(self) -> Dict[str, List[float]]:
        return self.metrics.history

    def train_model(
        self, dataset: SpotDataset, val_dataset: Optional[SpotDataset] = None
    ):
        """Execute the training loop for the model."""
        # Create data loaders - handles batch creation and shuffling
        train_loader = dataset.get_data_loader(shuffle=True)
        val_loader = val_dataset.get_data_loader(shuffle=False) if val_dataset else None

        # Create normalizer if needed - critical for consistent scaling across instances
        if self.model.normalizer is None:
            self.prepare_normalizer(dataset)

        self.metrics.print_header()

        # OneCycleLR scheduler helps with faster convergence and avoiding local minima
        steps_per_epoch = len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["max_learning_rate"],
            epochs=self.config["epochs"],
            steps_per_epoch=steps_per_epoch,
            pct_start=self.config["pct_start"],  # Percent of training to increase LR
            div_factor=self.config["div_factor"],  # Initial LR division factor
            final_div_factor=1e4,  # How much to reduce LR at the end
        )

        try:
            for epoch in range(self.config["epochs"]):
                start_epoch = datetime.now()

                train_loss, train_metrics = self._execute_training_phase(train_loader)
                epoch_duration = (datetime.now() - start_epoch).total_seconds()

                val_loss, val_metrics = None, None
                if val_loader:
                    val_loss, val_metrics = self._execute_validation_phase(val_loader)

                # Update metrics tracker and check for early stopping
                is_best, should_stop = self.metrics.update_epoch_state(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    train_metrics=train_metrics,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    epoch_duration=epoch_duration,
                )

                # Only save model when it improves to avoid disk usage
                if is_best:
                    self._save_model(val_loss if val_loss is not None else train_loss)

                self.metrics.print_epoch_stats()

                # Stop early if model isn't improving to save time
                if should_stop:
                    self.metrics.print_early_stopping()
                    break

        except KeyboardInterrupt:
            # Handle user interruption gracefully
            self._handle_interruption()
        except Exception as e:
            # Log any errors but still save metrics
            print(f"\nError during training: {str(e)}")
            self.metrics.save_to_files()
            raise

        self.metrics.save_to_files()

    def _execute_training_phase(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.train()
        epoch_loss = torch.tensor(0.0, device=self.device)
        epoch_metrics = {"mse": 0.0, "mape": 0.0, "direction": 0.0}
        batch_count = 0
        grad_norm = torch.tensor(0.0, device=self.device)

        # Use mixed precision for performance on compatible GPUs
        scaler = torch.GradScaler(enabled=torch.cuda.is_available())

        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for data, target in pbar:
                # Forward pass and loss calculation
                output = self.model(data)
                loss, metrics = self.criterion(output, target)

                # Backward pass with optimization
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)

                # Gradient clipping prevents exploding gradients in RNNs
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                # Optimizer step with mixed precision
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()

                # Accumulate metrics
                epoch_loss += loss
                self.metrics.track_batch_metrics(metrics, epoch_metrics)
                batch_count += 1

                # Update progress bar
                self.metrics.update_progress_bar(pbar, loss, metrics)

        # Calculate averages
        avg_loss, avg_metrics = self.metrics.compute_batch_average(
            epoch_loss.item(), epoch_metrics, batch_count
        )

        # Add learning rate to metrics for tracking
        current_lr = self.optimizer.param_groups[0]["lr"]
        avg_metrics["learning_rate"] = current_lr
        avg_metrics["grad_norm"] = grad_norm.item()

        return avg_loss, avg_metrics

    def _execute_validation_phase(
        self, val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Execute validation with no gradients to check for overfitting."""
        self.model.eval()
        total_loss = 0
        val_metrics = {"mse": 0.0, "mape": 0.0, "direction": 0.0}
        batch_count = 0

        # Determine if data is already on device (for speed optimization)
        sample_batch = next(iter(val_loader))
        data_on_device = sample_batch[0].device == self.device

        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device only if not already there
                if not data_on_device:
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                # Forward pass
                output = self.model(data)
                loss, metrics = self.criterion(output, target)

                total_loss += loss.item()
                self.metrics.track_batch_metrics(metrics, val_metrics)
                batch_count += 1

        # Calculate averages
        avg_loss, avg_metrics = self.metrics.compute_batch_average(
            total_loss, val_metrics, batch_count
        )

        return avg_loss, avg_metrics

    def _save_model(self, loss: float) -> None:
        """Save model with its normalizer to preserve the complete forecasting pipeline."""
        model_state = {
            "model_state_dict": self.model.state_dict(),
            "loss": loss,
            "config": self.config,
        }
        self.checkpoints.save_if_best(model_state)

    def _handle_interruption(self) -> None:
        """
        Handle user interruption gracefully by saving the current state.
        This ensures training progress isn't lost if the user needs to stop.
        """
        print("\nTraining interrupted by user")
        print("Saving current model state...")

        # Use the best available loss metric
        current_loss = (
            self.metrics._val_loss
            if self.metrics._val_loss is not None
            else self.metrics._train_loss
        )

        model_state = {
            "model_state_dict": self.model.state_dict(),
            "loss": current_loss,
            "config": self.config,
        }
        self.checkpoints.save(model_state, is_best=False)
        self.metrics.save_to_files()

    def _convert_numeric_params(self):
        """
        Convert config parameters to appropriate numeric types.
        Config parser returns strings, but we need actual numbers.
        """
        self.config["mse_weight"] = float(self.config["mse_weight"])
        self.config["learning_rate"] = float(self.config["learning_rate"])
        self.config["weight_decay"] = float(self.config["weight_decay"])
        self.config["max_learning_rate"] = float(self.config["max_learning_rate"])
        self.config["pct_start"] = float(self.config["pct_start"])
        self.config["div_factor"] = float(self.config["div_factor"])

        self.config["epochs"] = int(self.config["epochs"])
        self.config["patience"] = int(self.config["patience"])
