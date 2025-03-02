from typing import List, Dict, Tuple, DefaultDict, Optional, Any
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from dataset import SpotDataset
from utils import get_device, load_config
from model import Model


class Evaluate:
    REQUIRED_FIELDS = [
        "eval_step",
        "prediction_length",
        "n_timesteps_metrics",
        "batch_size",
    ]

    def __init__(
        self, model: Model, config_path: str = "config.yaml", output_dir: str = "output"
    ):
        """Initialize evaluation process for the model."""
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()  # Always evaluate in eval mode

        self.config = load_config(config_path, "evaluate_config", self.REQUIRED_FIELDS)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Batch size for efficient processing
        self.batch_size = self.config.get("batch_size", 32)

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
            "smape",
            "smape_std",
            "smape_cv",
            "rmse",
            "direction_accuracy",
        ]

    def get_prediction_results(self, id_instance):
        """Return prediction results for a specific instance."""
        return self.prediction_results[id_instance]

    def evaluate_instance(
        self, instance_id: int, dataset: SpotDataset
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a single instance using batch processing, keeping everything on GPU until final result.

        Only converts to CPU at the last moment when storing results.
        Processing in batches is critical for large datasets to:
        1. Reduce memory usage
        2. Better utilize GPU parallelism
        3. Avoid potential out-of-memory errors
        """
        try:
            # Get all sequences for this instance (already on GPU)
            instance_data = dataset.get_sequences(instance_id)
            if not instance_data or len(instance_data.get("sequences", [])) == 0:
                return []

            # Keep as tensors on GPU
            sequences = instance_data["sequences"]
            targets = instance_data["targets"]

            # Process in batches for memory efficiency
            num_sequences = sequences.shape[0]
            all_predictions = []

            # Process sequences in batches without unnecessary conversions
            for i in range(0, num_sequences, self.batch_size):
                batch_end = min(i + self.batch_size, num_sequences)
                batch_sequences = sequences[i:batch_end]  # Already on GPU

                # Get predictions (stays on GPU)
                with torch.no_grad():
                    batch_predictions = self.model.forecast(
                        batch_sequences, self.config["prediction_length"], instance_id
                    )

                all_predictions.append(batch_predictions)

            if not all_predictions:
                return []

            # Combine predictions into single tensor
            predictions = torch.cat(all_predictions, dim=0)

            # For final storage, keep as tensors
            return [(targets[i], predictions[i]) for i in range(num_sequences)]

        except Exception as e:
            print(f"Error evaluating instance {instance_id}: {str(e)}")
            return []

    def evaluate_all(self, dataset: SpotDataset) -> Dict:
        """
        Evaluate all instances in the dataset using efficient batch processing.

        Returns metrics by instance ID to enable both aggregated and
        per-instance analysis of model performance.
        """
        print("Evaluation Configuration:")
        print(f"- Model: {type(self.model).__name__}")
        print(f"- Prediction length: {self.config['prediction_length']}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Total instances: {len(dataset.instance_ids)}\n")

        # Verify that the model has a normalizer attached
        if self.model.normalizer is not None:
            print("Using model's normalizer for evaluation")
            norm_stats = self.model.normalizer.get_params_summary()
            print(f"- Normalizer has parameters for {norm_stats['count']} instances")
        else:
            # Missing normalizer is a serious issue worth highlighting
            print("Warning: Model doesn't have a normalizer attached")

        # Process each instance sequentially
        with tqdm(total=len(dataset.instance_ids), desc="Processing instances") as pbar:
            for instance_id in dataset.instance_ids:
                results = self.evaluate_instance(instance_id, dataset)

                if results:
                    # Store results and calculate metrics
                    self.prediction_results[instance_id] = results
                    list_target = [r[0] for r in results]
                    list_pred = [r[1] for r in results]
                    self.segmented_metrics[instance_id] = self._calculate_metrics(
                        list_pred, list_target, self.config["n_timesteps_metrics"]
                    )
                else:
                    self.failed_instances.append(instance_id)

                pbar.update(1)

        if self.failed_instances:
            # Report failures to help identify data issues
            print(
                f"\nWarning: {len(self.failed_instances)} instances had insufficient data for evaluation"
            )

        # Save metrics to enable offline analysis
        self._save_metrics()

        return self.segmented_metrics

    def _save_metrics(self) -> None:
        """
        Save metrics to JSON files for external analysis and reporting.

        The separation into overall and instance-specific files helps with
        both aggregate analysis and detailed examination of problematic instances.
        """
        # Convert predictions and targets to CPU NumPy only when saving to disk
        result_dict = {}

        for instance_id, results in self.prediction_results.items():
            # Convert tensors to CPU then NumPy arrays for JSON serialization
            result_dict[str(instance_id)] = [
                (target.cpu().numpy().tolist(), pred.cpu().numpy().tolist())
                for target, pred in results
            ]

        # Save prediction results
        with open(os.path.join(self.output_dir, "predictions.json"), "w") as f:
            json.dump(result_dict, f)

        # Save metrics (already in correct format)
        overall_metrics = self._calculate_overall_metrics()
        with open(os.path.join(self.output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(overall_metrics, f, indent=2)

        # Save per-instance metrics (convert keys to strings for JSON)
        with open(os.path.join(self.output_dir, "instance_metrics.json"), "w") as f:
            json.dump(
                {str(k): v for k, v in self.segmented_metrics.items()}, f, indent=2
            )

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all instances.

        This provides a single set of performance numbers for the entire model,
        while still capturing statistical distribution through min/max/std values.
        """
        if not self.segmented_metrics:
            return {}

        all_metrics = {}

        # Calculate statistics across all instances
        for metric in self.metrics:
            if metric == "n_timestep":
                continue

            values = []
            for instance_id in self.segmented_metrics:
                for segment in self.segmented_metrics[instance_id]:
                    if metric in segment:
                        values.append(segment[metric])

            if values:
                # Store mean, std, min, max to understand the distribution
                all_metrics[f"avg_{metric}"] = float(np.mean(values))
                all_metrics[f"std_{metric}"] = float(np.std(values))
                all_metrics[f"min_{metric}"] = float(np.min(values))
                all_metrics[f"max_{metric}"] = float(np.max(values))

        # Add summary counts
        all_metrics["total_instances"] = len(self.segmented_metrics)
        all_metrics["failed_instances"] = len(self.failed_instances)

        return all_metrics

    @staticmethod
    def _calculate_metrics(
        predictions: List[torch.Tensor], targets: List[torch.Tensor], n_timesteps: int
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
        # Stack to create batched tensors
        pred_tensor = torch.stack(predictions)
        target_tensor = torch.stack(targets)

        n_segments = pred_tensor.shape[1] // n_timesteps
        metrics = []

        # Small epsilon to avoid division by zero
        epsilon = 1e-10

        for i in range(n_segments):
            start = i * n_timesteps
            end = start + n_timesteps

            pred_segments = pred_tensor[:, start:end]
            target_segments = target_tensor[:, start:end]

            # Calculate absolute differences
            abs_diff = torch.abs(pred_segments - target_segments)
            abs_sum = torch.abs(pred_segments) + torch.abs(target_segments)

            # Avoid division by zero
            abs_sum = torch.clamp_min(abs_sum, epsilon)
            abs_targets = torch.clamp_min(torch.abs(target_segments), epsilon)

            # SMAPE - symmetric mean absolute percentage error
            smape_values = 2 * abs_diff / abs_sum

            # MAPE - mean absolute percentage error
            mape_values = abs_diff / abs_targets

            # Direction accuracy - compares signs of price movements
            pred_diff = pred_segments[:, 1:] - pred_segments[:, :-1]
            target_diff = target_segments[:, 1:] - target_segments[:, :-1]
            direction_match = (torch.sign(pred_diff) == torch.sign(target_diff)).float()

            # Convert results to Python types for JSON serialization
            metrics.append(
                {
                    "n_timestep": start,
                    "mape": float(torch.mean(mape_values).item() * 100),
                    "smape": float(torch.mean(smape_values).item() * 100),
                    "smape_std": float(torch.std(smape_values).item() * 100),
                    "smape_cv": float(
                        (
                            torch.std(smape_values)
                            / (torch.mean(smape_values) + epsilon)
                        ).item()
                        * 100
                    ),
                    "rmse": float(
                        torch.sqrt(
                            torch.mean((pred_segments - target_segments) ** 2)
                        ).item()
                    ),
                    "direction_accuracy": float(
                        torch.mean(direction_match).item() * 100
                    ),
                }
            )

        return metrics
