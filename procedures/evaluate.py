from typing import List, Dict, Tuple, DefaultDict, Optional, Any
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from utils import get_device, load_config
from model import Model


class Evaluate:
    FIELDS_EVALUATE = [
        "prediction_length",
        "n_timesteps_metrics",
        "batch_size",
    ]
    FIELDS_DATASET = ["sequence_length", "window_step", "prediction_length"]

    def __init__(self, model: Model, work_dir: str):
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

        self.config = load_config(
            f"{work_dir}/config.yaml", "evaluate_config", self.FIELDS_EVALUATE
        )
        self.dataset_config = load_config(
            f"{work_dir}/config.yaml", "dataset_config", self.FIELDS_DATASET
        )
        self.output_dir = f"{work_dir}/evaluation"
        os.makedirs(self.output_dir, exist_ok=True)

        # Batch size for efficient processing
        self.batch_size = self.config.get("batch_size", 32)
        self.eval_prediction_length = (
            self.config["prediction_length"] * self.dataset_config["prediction_length"]
        )

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

    def _create_sequences(
        self, instance_df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for evaluation using rolling window approach.

        Args:
            instance_df: DataFrame for a single instance

        Returns:
            Tuple of (input_sequences, target_sequences) as tensors
        """
        sequence_length = self.dataset_config["sequence_length"]
        window_step = self.dataset_config["window_step"]

        values = instance_df["spot_price"].values.astype(np.float32)
        total_length = len(values)

        # Calculate how many sequences we can create
        required_length = sequence_length + self.eval_prediction_length
        if total_length < required_length:
            return None, None  # type: ignore

        n_sequences = (total_length - required_length) // window_step + 1

        # Create sequences and targets
        input_sequences = []
        target_sequences = []

        for i in range(n_sequences):
            start_idx = i * window_step
            end_input_idx = start_idx + sequence_length
            end_target_idx = end_input_idx + self.eval_prediction_length

            if end_target_idx > total_length:
                break

            input_seq = values[start_idx:end_input_idx]
            target_seq = values[end_input_idx:end_target_idx]

            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

        if not input_sequences:
            return None, None  # type: ignore

        # Convert to tensors
        input_sequences = np.array(input_sequences)
        target_sequences = np.array(target_sequences)
        input_tensor = torch.tensor(
            input_sequences, dtype=torch.float32, device=self.device
        )
        target_tensor = torch.tensor(
            target_sequences, dtype=torch.float32, device=self.device
        )

        # Add feature dimension to input
        input_tensor = input_tensor.unsqueeze(-1)

        return input_tensor, target_tensor

    def evaluate_instance(
        self, instance_id: int, test_df: pd.DataFrame
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a single instance using the test DataFrame directly.

        Args:
            instance_id: ID of the instance to evaluate
            test_df: DataFrame containing test data

        Returns:
            List of (target, prediction) tensor pairs
        """
        try:
            instance_df = test_df[test_df["id_instance"] == instance_id]

            if len(instance_df) == 0:
                print(f"Warning: No data found for instance {instance_id}")
                return []

            input_sequences, target_sequences = self._create_sequences(instance_df)

            if input_sequences is None or target_sequences is None:
                print(
                    f"Warning: Insufficient data for instance {instance_id} to create sequences"
                )
                return []

            all_predictions = []
            num_sequences = input_sequences.shape[0]

            for i in range(0, num_sequences, self.batch_size):
                batch_end = min(i + self.batch_size, num_sequences)
                batch_inputs = input_sequences[i:batch_end]

                with torch.no_grad():
                    batch_predictions = self.model.forecast(
                        batch_inputs,
                        self.eval_prediction_length,
                        [instance_id] * len(batch_inputs),
                    )

                all_predictions.append(batch_predictions)

            if not all_predictions:
                return []

            # Combine predictions into single tensor
            predictions = torch.cat(all_predictions, dim=0)

            # Return pairs of targets and predictions
            return [(target_sequences[i], predictions[i]) for i in range(num_sequences)]

        except Exception as e:
            print(f"Error evaluating instance {instance_id}: {str(e)}")
            return []

    def evaluate_all(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate all instances in the test DataFrame.

        Args:
            test_df: DataFrame containing test data

        Returns:
            Dictionary of metrics by instance ID
        """
        print("Evaluation Configuration:")
        print(f"- Model: {type(self.model).__name__}")
        print(f"- Input sequence length: {self.dataset_config['sequence_length']}")
        print(f"- Prediction length: {self.config['prediction_length']}")
        print(f"- Window step: {self.dataset_config['window_step']}")
        print(f"- Batch size: {self.batch_size}")

        # Get unique instance IDs from the data
        instance_ids = test_df["id_instance"].unique()
        print(f"- Total instances: {len(instance_ids)}\n")

        if self.model.normalizer is not None:
            norm_stats = self.model.normalizer.get_params_summary()
            print(
                f"Using model's normalizer for evaluation {norm_stats['count']} instances"
            )
        else:
            print("Warning: Model doesn't have a normalizer attached")

        with tqdm(total=len(instance_ids), desc="Processing instances") as pbar:
            for instance_id in instance_ids:
                results = self.evaluate_instance(instance_id, test_df)

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
        with open(os.path.join(self.output_dir, "overall_metrics.json"), "w") as f:
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
