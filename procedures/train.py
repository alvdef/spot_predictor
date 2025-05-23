from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import get_loss
from utils import get_device, load_config, MetricsTracker, get_logger
from dataset import SpotDataset, Normalizer
from model import Model


class Training:
    REQUIRED_CONFIG_FIELDS = [
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
        work_dir: str,
        instance_features_df: Optional[pd.DataFrame] = None,
    ):
        """Initialize training process for the model.

        Args:
            model: The time series model to train
            work_dir: Directory for saving model checkpoints and metrics
            instance_features_df: Optional dataframe containing instance features
        """
        self.logger = get_logger(__name__)
        self.device = get_device()
        self.model = model
        self.work_dir = work_dir
        self.instance_features_df = instance_features_df

        self.metrics = MetricsTracker(work_dir + "/training")
        resumed_training = self.metrics.load()
        if resumed_training:
            self.logger.info("Resumed training. Loading model...")
            self.model.load()

        self.config = load_config(
            f"{work_dir}/config.yaml",
            "training_hyperparams",
            self.REQUIRED_CONFIG_FIELDS,
        )

        self._convert_numeric_params()

        self.metrics.early_stopping_patience = self.config["patience"]

        # Initialize loss function with configurable weights
        self.criterion = get_loss(work_dir)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999),  # Default betas work well for most time series tasks
        )

        if resumed_training and self.metrics._learning_rate > 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.metrics._learning_rate
            self.logger.info(
                f"Restored learning rate to {self.metrics._learning_rate:.6e}"
            )

    def prepare_normalizer(self, dataset: SpotDataset) -> None:
        """
        Create and fit normalizer using all available data for each instance.
        Uses the full sequence data to compute more stable normalization parameters.

        The normalizer is fitted per-instance but will be used with lists of instance IDs
        during training and inference to maximize batch processing efficiency.
        """
        self.logger.info("Preparing normalizer for the model...")

        normalizer = Normalizer()

        unique_ids = list(set(dataset.get_sequences()["instance_ids"]))
        for instance_id in unique_ids:
            # Use the instance_id parameter of get_sequences to get data for a specific instance
            instance_data = dataset.get_sequences(instance_id=instance_id)

            price_values = instance_data["sequences"].reshape(-1)
            normalizer.fit(instance_id, price_values)

        # Compute global statistics for default normalization
        normalizer.compute_global_stats()

        stats = normalizer.get_params_summary()
        self.logger.info(f"Normalizer prepared for {stats['count']} instances")
        self.logger.info(
            f"Global normalization mean: {normalizer.global_stats['mean']:.4f}, std: {normalizer.global_stats['std']:.4f}"
        )

        self.model.attach_normalizer(normalizer)

    @property
    def history(self) -> Dict[str, List[float]]:
        return self.metrics.history

    def train_model(
        self, dataset: SpotDataset, val_dataset: Optional[SpotDataset] = None
    ):
        """Execute the training loop for the model."""
        start_epoch = self.metrics._epoch
        remaining_epochs = self.config["epochs"] - start_epoch

        self.logger.info("Training Configuration:")
        self.logger.info(f"- Model: {type(self.model).__name__}")
        self.logger.info(
            f"- Input sequence length: {dataset.config['sequence_length']}"
        )
        self.logger.info(f"- Prediction length: {dataset.config['prediction_length']}")
        self.logger.info(f"- Window step: {dataset.config['window_step']}")
        self.logger.info(f"- Batch size: {dataset.config['batch_size']}")
        self.logger.info(f"- Learning rate: {self.config['learning_rate']}")
        self.logger.info(f"- Max learning rate: {self.config['max_learning_rate']}")
        self.logger.info(f"- Weight decay: {self.config['weight_decay']}")
        if start_epoch > 0:
            self.logger.info(f"- Starting from epoch: {start_epoch}")
            self.logger.info(f"- Epochs left: {remaining_epochs}")
        else:
            self.logger.info(f"- Epochs: {remaining_epochs}")
        self.logger.info(f"- Early stopping patience: {self.config['patience']}")
        self.logger.info(f"- Optimizer: {type(self.optimizer).__name__}")
        self.logger.info(f"- Loss function: {type(self.criterion).__name__}")

        # Create data loaders - handles batch creation and shuffling
        train_loader = dataset.get_data_loader(shuffle=True)
        val_loader = val_dataset.get_data_loader(shuffle=False) if val_dataset else None

        if self.model.normalizer is None:
            self.prepare_normalizer(dataset)

        metric_names = self.criterion.get_metric_names()
        self.metrics.print_header(metrics_keys=metric_names)

        # OneCycleLR scheduler helps with faster convergence and avoiding local minima
        steps_per_epoch = len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["max_learning_rate"],
            epochs=remaining_epochs,
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

                if is_best:
                    self.model.save()

                self.metrics.print_epoch_stats()

                # Stop early if model isn't improving to save time
                if should_stop:
                    self.metrics.print_early_stopping()
                    self.logger.info("Early stopping triggered - halting training")
                    break

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

        finally:
            # Always save the current state regardless of how training ended
            self.logger.info("Saving current model state and metrics...")
            self.model.save()
            self.metrics.save()

    def _execute_training_phase(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.train()
        epoch_loss = torch.tensor(0.0, device=self.device)
        epoch_metrics = defaultdict(float)
        batch_count = 0
        grad_norm = torch.tensor(0.0, device=self.device)

        scaler = torch.GradScaler()

        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for data, target in pbar:
                output = self.model(data, target)
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
        val_metrics = defaultdict(float)
        batch_count = 0

        with torch.no_grad():
            for data, target in val_loader:
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

    def _convert_numeric_params(self):
        """
        Convert config parameters to appropriate numeric types.
        Config parser returns strings, but we need actual numbers.
        """
        self.config["learning_rate"] = float(self.config["learning_rate"])
        self.config["weight_decay"] = float(self.config["weight_decay"])
        self.config["max_learning_rate"] = float(self.config["max_learning_rate"])
        self.config["pct_start"] = float(self.config["pct_start"])
        self.config["div_factor"] = float(self.config["div_factor"])

        self.config["epochs"] = int(self.config["epochs"])
        self.config["patience"] = int(self.config["patience"])
