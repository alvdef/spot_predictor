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

from utils import get_device, load_config
from utils.trend_metrics import (
    calculate_significant_trend_accuracy,
    calculate_spot_price_savings,
    calculate_perfect_information_savings,
)
from model import Model


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

    def __init__(self, model: Model, work_dir: str):
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

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
        sequence_length = self.config["sequence_length"]
        window_step = self.config["window_step"]

        values = instance_df["spot_price"].values.astype(np.float32)
        total_length = len(values)

        # Calculate how many sequences we can create
        required_length = sequence_length + self.config["prediction_length"]
        if total_length < required_length:
            return None, None  # type: ignore

        n_sequences = (total_length - required_length) // window_step + 1

        # Create sequences and targets
        input_sequences = []
        target_sequences = []

        for i in range(n_sequences):
            start_idx = i * window_step
            end_input_idx = start_idx + sequence_length
            end_target_idx = end_input_idx + self.config["prediction_length"]

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
                        self.config["prediction_length"],
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

    def evaluate_all(self, test_df: pd.DataFrame, instance_info_df=None) -> Dict:
        """
        Evaluate all instances in the test DataFrame.

        Args:
            test_df: DataFrame containing test data

        Returns:
            Dictionary of metrics by instance ID
        """
        print("Evaluation Configuration:")
        print(f"- Model: {type(self.model).__name__}")
        print(f"- Input sequence length: {self.config['sequence_length']}")
        print(f"- Prediction length: {self.config['prediction_length']}")
        print(f"- Window step: {self.config['window_step']}")
        print(f"- Batch size: {self.batch_size}")

        # Get unique instance IDs from the data
        instance_ids = test_df["id_instance"].unique()
        print(f"- Total instances to evaluate: {len(instance_ids)}\n")

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
        overall_metrics = self._calculate_overall_metrics()
        self.overall_metrics = overall_metrics
        self._save_metrics(self.output_dir, instance_info_df)

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

            # Convert any numpy or torch types to Python native types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return convert_to_serializable(obj.tolist())
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {
                        key: convert_to_serializable(value)
                        for key, value in obj.items()
                    }
                elif isinstance(obj, date):
                    return obj.isoformat()
                else:
                    return obj

            return convert_to_serializable(full_config)

        except Exception as e:
            print(f"Warning: Could not load full config: {str(e)}")
            return {}

    def _save_metrics(self, output_dir, instance_info_df=None):
        """Save metrics to JSON files."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Convert int64 keys to strings for JSON serialization
        instance_metrics_string_keys = {
            str(k): v for k, v in self.segmented_metrics.items()
        }

        # Save per-instance metrics
        instance_metrics_file = os.path.join(output_dir, "instance_metrics.json")
        with open(instance_metrics_file, "w") as f:
            json.dump(instance_metrics_string_keys, f, indent=2)

        # Save overall metrics
        overall_metrics_file = os.path.join(output_dir, "overall_metrics.json")
        with open(overall_metrics_file, "w") as f:
            json.dump(self.overall_metrics, f, indent=2)

        # Create new combined metrics file if instance_info_df is provided
        if instance_info_df is not None:
            # Get the full configuration
            work_dir = os.path.dirname(output_dir)
            full_config = self._load_full_config(work_dir)

            combined_metrics = {
                "overall_metrics": self.overall_metrics,
                "config": full_config,  # Add configuration to the output
                "instances": {},
            }

            # For each instance, add metadata and metrics
            for instance_id, metrics_list in self.segmented_metrics.items():
                try:
                    # Get instance properties
                    instance_props = instance_info_df.loc[instance_id].to_dict()

                    # Add to combined metrics
                    combined_metrics["instances"][str(instance_id)] = {
                        "metadata": {"instance_id": str(instance_id), **instance_props},
                        "metrics": metrics_list,
                    }
                except KeyError:
                    # Skip if instance not found in instance_info_df
                    print(
                        f"Warning: Instance {instance_id} not found in instance_info_df"
                    )

            # Save combined JSON file
            combined_file = os.path.join(output_dir, "dashboard_metrics.json")
            with open(combined_file, "w") as f:
                json.dump(combined_metrics, f, indent=2)

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
        # Stack to create batched tensors
        pred_tensor = torch.stack(predictions)
        target_tensor = torch.stack(targets)

        # Calculate whole-sequence metrics before segmentation
        # These metrics evaluate the entire prediction, not just windows
        sig_accuracy = calculate_significant_trend_accuracy(
            pred_tensor, target_tensor, significance_threshold
        )
        cost_savings = calculate_spot_price_savings(
            pred_tensor, target_tensor, decision_window
        )
        # Calculate the perfect information savings (theoretical maximum)
        perfect_savings = calculate_perfect_information_savings(
            target_tensor, decision_window
        )

        # Calculate savings efficiency (what percentage of perfect savings is achieved)
        if perfect_savings > 0:
            savings_efficiency = (cost_savings / perfect_savings) * 100
        else:
            savings_efficiency = 100.0 if cost_savings == 0 else 0.0

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
            abs_targets = torch.clamp_min(torch.abs(target_segments), epsilon)

            # MAPE - mean absolute percentage error
            mape_values = abs_diff / abs_targets

            # Convert results to Python types for JSON serialization
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
