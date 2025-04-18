from typing import List, Dict, Tuple, DefaultDict, Optional, Any
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import yaml
from datetime import date
from torch.utils.data import Dataset, DataLoader, Subset

from utils import (
    calculate_significant_trend_accuracy,
    calculate_spot_price_savings,
    calculate_perfect_information_savings,
    get_logger,
    get_device,
    load_config,
)
from model import Model
from dataset import SpotDataset


class Evaluate:
    REQUIRED_CONFIG_FIELDS = [
        "prediction_length",
        "n_timesteps_metrics",
        "batch_size",
        "significance_threshold",
        "sequence_length",
        "window_step",
        "prediction_length",
    ]

    def __init__(self, model: Model, dataset: SpotDataset, work_dir: str):
        """
        Initialize the evaluation framework.

        Args:
            model: The model to evaluate
            work_dir: Working directory containing config and for storing results
            dataset: Optional SpotDataset to use for evaluation. If not provided,
                     one will need to be passed to evaluate_all.
        """
        self.logger = get_logger(__name__)
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()
        self.work_dir = work_dir
        self.dataset = dataset

        self.config = load_config(
            f"{work_dir}/config.yaml", "evaluate_config", self.REQUIRED_CONFIG_FIELDS
        )
        self.output_dir = f"{work_dir}/evaluation"
        os.makedirs(self.output_dir, exist_ok=True)

        # Batch size for efficient processing
        self.batch_size = self.config["batch_size"]

        # Storage for results and metrics
        self.segmented_metrics: DefaultDict[int, List[Dict]] = defaultdict(list)
        self.prediction_results: DefaultDict[
            int, List[Tuple[torch.Tensor, torch.Tensor]]
        ] = defaultdict(list)
        self.failed_instances: List[int] = []

    @property
    def metrics(self):
        """Metrics used to evaluate forecast quality."""
        return [
            "n_timestep",
            "mape",
            "rmse",
            "sgnif_trend_acc",
            "cost_savings",
            "perfect_savings",
            "savings_efficiency",
        ]

    def get_prediction_results(self, id_instance):
        """Return prediction results for a specific instance."""
        return self.prediction_results[id_instance]

    def evaluate_instance(
        self, instance_id: int, instance_loader: DataLoader
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a single instance using DataLoader.

        Args:
            instance_id: ID of the instance to evaluate
            test_dataset: DataLoader containing test data

        Returns:
            List of (target, prediction) tensor pairs
        """
        try:
            all_targets = []
            all_predictions = []

            for inputs, targets in instance_loader:
                predictions = self.model.forecast(
                    inputs,
                    self.config["prediction_length"],
                    [instance_id] * self.batch_size,
                )

                all_targets.append(targets)
                all_predictions.append(predictions)

            if not all_predictions:
                return []

            targets = torch.cat(all_targets, dim=0)
            predictions = torch.cat(all_predictions, dim=0)

            # Return pairs of targets and predictions
            return [(targets[i], predictions[i]) for i in range(len(targets))]

        except ValueError as e:
            self.logger.warning(f"No data found for instance {instance_id}: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(
                f"Error evaluating instance {instance_id}: {str(e)}", exc_info=True
            )
            return []

    def evaluate_all(self) -> Dict:
        """
        Evaluate all instances in the provided dataset.

        Args:
            dataset: SpotDataset containing test data. If not provided,
                    the dataset passed during initialization will be used.

        Returns:
            Dictionary of metrics by instance ID

        Raises:
            ValueError: If no dataset is available
        """
        self.logger.info("Starting model evaluation")
        self.logger.info(f"- Model type: {type(self.model).__name__}")
        self.logger.info(f"- Input sequence length: {self.config['sequence_length']}")
        self.logger.info(f"- Prediction length: {self.config['prediction_length']}")
        self.logger.info(f"- Window step: {self.config['window_step']}")
        self.logger.info(f"- Batch size: {self.batch_size}")

        # Get unique instance IDs from the dataset using the new method
        instance_ids = self.dataset.get_unique_instance_ids()
        self.logger.info(f"- Total instances to evaluate: {len(instance_ids)}")

        if self.model.normalizer is not None:
            norm_stats = self.model.normalizer.get_params_summary()
            self.logger.info(
                f"- Total instances in model's normalizer: {norm_stats['count']}"
            )
        else:
            self.logger.warning("Model doesn't have a normalizer attached")

        with tqdm(total=len(instance_ids), desc="Processing instances") as pbar:
            for instance_id in instance_ids:
                instance_loader = self.dataset.get_instance_dataloader(
                    instance_id, batch_size=self.batch_size, shuffle=False
                )
                results = self.evaluate_instance(instance_id, instance_loader)

                if results:
                    self.prediction_results[instance_id] = results
                    list_target = [r[0] for r in results]
                    list_pred = [r[1] for r in results]
                    self.segmented_metrics[instance_id] = self._calculate_metrics(
                        list_pred, list_target, self.config["n_timesteps_metrics"]
                    )
                else:
                    self.logger.warning(f"No results for instance {instance_id}")
                    self.failed_instances.append(instance_id)

                pbar.update(1)

            self.logger.info(
                f"{len(self.failed_instances)} instances had insufficient data for evaluation"
            )

        return self.segmented_metrics

    def _load_full_config(self, work_dir: str) -> Dict[str, Any]:
        """
        Load the full configuration from config.yaml to include in the output metrics.

        Args:
            work_dir: Working directory containing the config file

        Returns:
            Dictionary containing all configuration sections
        """
        config_path = os.path.join(work_dir, "config.yaml")
        try:
            with open(config_path, "r") as file:
                full_config = yaml.safe_load(file)

            for section in full_config.values():
                if isinstance(section, dict):
                    for key, value in section.items():
                        if isinstance(value, date):
                            section[key] = value.isoformat()

            return full_config

        except Exception as e:
            self.logger.warning(f"Could not load full config: {str(e)}")
            return {}

    def save_metrics(self):
        instance_metrics_file = os.path.join(self.output_dir, "instance_metrics.json")
        with open(instance_metrics_file, "w") as f:
            instance_metrics_string_keys = {
                str(k): v for k, v in self.segmented_metrics.items()
            }
            json.dump(instance_metrics_string_keys, f, indent=2)

        combined_metrics = {
            "config": self._load_full_config(self.work_dir),
            "instances": {},
        }

        instance_info_df = self.dataset.instance_features_df

        for instance_id, metrics_list in self.segmented_metrics.items():
            instance_props = (
                instance_info_df.loc[instance_id].to_dict()
                if instance_id in instance_info_df.index
                else {}
            )

            combined_metrics["instances"][str(instance_id)] = {
                "metadata": {"instance_id": str(instance_id), **instance_props},
                "metrics": metrics_list,
            }

        combined_file = os.path.join(self.output_dir, "dashboard_metrics.json")
        with open(combined_file, "w") as f:
            json.dump(combined_metrics, f, indent=2)

        self.logger.info(f"Metrics saved to {self.output_dir}")

    def _calculate_metrics(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_timesteps: int,
    ) -> List[Dict]:
        """
        Calculate evaluation metrics using PyTorch operations for better efficiency.

        Args:
            predictions: List of prediction tensors
            targets: List of target tensors
            n_timesteps: Window size for segmented metrics

        Returns:
            List of metric dictionaries for each segment
        """
        significance_threshold = self.config["significance_threshold"]
        decision_window = self.config["prediction_length"]

        pred_tensor = torch.stack(predictions)
        target_tensor = torch.stack(targets)

        sig_accuracy = calculate_significant_trend_accuracy(
            pred_tensor, target_tensor, significance_threshold
        )
        cost_savings = calculate_spot_price_savings(
            pred_tensor, target_tensor, decision_window
        )
        perfect_savings = calculate_perfect_information_savings(
            target_tensor, decision_window
        )

        if perfect_savings > 0:
            savings_efficiency = (cost_savings / perfect_savings) * 100
        else:
            savings_efficiency = 100.0 if cost_savings == 0 else 0.0

        n_segments = pred_tensor.shape[1] // n_timesteps
        metrics = []

        for i in range(n_segments):
            start = i * n_timesteps
            end = start + n_timesteps
            pred_segments = pred_tensor[:, start:end]
            target_segments = target_tensor[:, start:end]

            abs_diff = torch.abs(pred_segments - target_segments)
            abs_targets = torch.clamp_min(torch.abs(target_segments), 1e-10)

            mape_values = abs_diff / abs_targets

            segment_metrics = {
                "n_timestep": start,
                "mape": float(torch.mean(mape_values).item() * 100),
                "mse": float(torch.mean((pred_segments - target_segments) ** 2).item()),
                "sgnif_trend_acc": sig_accuracy,
                "cost_savings": cost_savings,
                "perfect_savings": perfect_savings,
                "savings_efficiency": float(savings_efficiency),
            }

            metrics.append(segment_metrics)

        return metrics
