import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from collections import defaultdict

from utils import get_device, load_config
from .model import Model


class Evaluate:
    REQUIRED_FIELDS = [
        "sequence_length",
        "eval_step",
        "prediction_length",
        "n_timesteps",
        "batch_size",
    ]

    def __init__(self, model: Model, config_path: str = "config.yaml", output_dir: str = "output"):
        self.model = model
        self.device = get_device()

        self.config = load_config(config_path, "evaluate_config", self.REQUIRED_FIELDS)
        os.makedirs(output_dir, exist_ok=True)

        self.general_metrics = defaultdict(list)
        self.segmented_metrics = defaultdict(list)
        self.prediction_results = defaultdict(list)
        self.failed_instances = []  # Track failed instances

    def get_failed_instances(self):
        """Get list of instance IDs that failed evaluation"""
        return self.failed_instances

    def evaluate_batch(self, instances_data: List[np.ndarray], batch_ids: List[int]):
        """Evaluate multiple instances in batches"""
        all_results = []
        required_len = self.config["sequence_length"] + self.config["prediction_length"]

        # Filter instances with correct length and pad if necessary
        valid_data = []
        valid_ids = []
        max_len = max(len(inst_data) for inst_data in instances_data)

        for inst_data, inst_id in zip(instances_data, batch_ids):
            if len(inst_data) >= required_len:
                # Pad sequence to max_len if needed
                if len(inst_data) < max_len:
                    padded = np.pad(
                        inst_data,
                        (0, max_len - len(inst_data)),
                        mode="constant",
                        constant_values=inst_data[-1],
                    )
                    valid_data.append(padded)
                else:
                    valid_data.append(inst_data)
                valid_ids.append(inst_id)
            else:
                self.failed_instances.append(inst_id)

        if not valid_data:
            return []

        batch_data = np.array(valid_data)
        num_experiments = (batch_data.shape[1] - required_len) // self.config[
            "eval_step"
        ] + 1
        batch_results = [[] for _ in range(len(valid_data))]

        for step in range(num_experiments):
            start_idx = step * self.config["eval_step"]
            end_idx = (
                start_idx
                + self.config["sequence_length"]
                + self.config["prediction_length"]
            )

            if end_idx > batch_data.shape[1]:
                break

            input_seq = batch_data[
                :, start_idx : start_idx + self.config["sequence_length"]
            ]
            target = batch_data[:, start_idx + self.config["sequence_length"] : end_idx]

            input_tensor = torch.FloatTensor(input_seq).to(self.device)
            with torch.no_grad():
                predictions = self.model.forecast(
                    input_tensor, self.config["prediction_length"]
                )
                predictions = predictions.cpu().numpy()

            for j in range(len(valid_data)):
                batch_results[j].append((target[j], predictions[j]))

        all_results.extend(batch_results)
        return all_results

    def evaluate_all(self, df: pd.DataFrame):
        """Evaluate all instances in batches"""
        print("\nEvaluation Configuration:")
        print(f"- Sequence length: {self.config['sequence_length']}")
        print(f"- Prediction length: {self.config['prediction_length']}")
        print(f"- Total instances: {df['id_instance'].nunique()}")

        self.segmented_metrics.clear()
        self.failed_instances = []  # Reset failed instances
        grouped_data = []
        instance_ids = []

        for instance_id, group in df.groupby("id_instance"):
            grouped_data.append(group["spot_price"].values)
            instance_ids.append(instance_id)

        batch_size = self.config["batch_size"]
        with tqdm(total=len(grouped_data), desc="Evaluating instances") as pbar:
            for i in range(0, len(grouped_data), batch_size):
                batch_data = grouped_data[i : i + batch_size]
                batch_ids = instance_ids[i : i + batch_size]

                results = self.evaluate_batch(batch_data, batch_ids)

                if results:  # Only process if we have valid results
                    for instance_id, instance_results in zip(batch_ids, results):
                        if instance_results:
                            predictions, targets = zip(*instance_results)
                            self.segmented_metrics[instance_id] = (
                                self._calculate_metrics(
                                    predictions, targets, self.config["n_timesteps"]
                                )
                            )
                pbar.update(len(batch_data))

        print(f"\nCompleted evaluation of {len(self.segmented_metrics)} instances")
        if self.failed_instances:
            print(f"Failed instances: {len(self.failed_instances)}")
        return self.segmented_metrics

    @staticmethod
    def _calculate_metrics(
        predictions: List[np.ndarray], targets: List[np.ndarray], n_timesteps: int
    ):
        predictions_array = np.array(predictions)
        targets_array = np.array(targets)
        
        num_timesteps_agg = len(predictions[0]) / n_timesteps
        timesteps_metrics = []
        
        for n in range(int(num_timesteps_agg)):
            start_idx = int(n * num_timesteps_agg)
            end_idx = int((n + 1) * num_timesteps_agg)
            
            pred_segment = predictions_array[:, start_idx:end_idx]
            target_segment = targets_array[:, start_idx:end_idx]
            
            mse = np.mean((pred_segment - target_segment) ** 2, axis=1)
            mape = np.mean(np.abs((target_segment - pred_segment) / target_segment), axis=1) * 100
            smape = np.mean(2 * np.abs(pred_segment - target_segment) / 
                          (np.abs(pred_segment) + np.abs(target_segment)), axis=1) * 100
            smape_std = np.std(2 * np.abs(pred_segment - target_segment) / 
                          (np.abs(pred_segment) + np.abs(target_segment)), axis=1) * 100
            # Compute coefficient of variation for SMAPE (CV = std/mean)
            avg_smape = np.mean(smape)
            cv_smape = (np.mean(smape_std) / avg_smape) if avg_smape != 0 else 0
            
            pred_diff = np.diff(pred_segment, axis=1)
            target_diff = np.diff(target_segment, axis=1)
            direction_correct = np.mean(np.sign(pred_diff) == np.sign(target_diff), axis=1)
            
            timestep_metric = {
                "n_timestep": start_idx,
                "mape": np.mean(mape),
                "smape": avg_smape,
                "smape_std": np.mean(smape_std),
                "smape_cv": cv_smape,  # added normalized consistency metric
                "rmse": np.sqrt(np.mean(mse)),
                "direction_accuracy": np.mean(direction_correct),
            }
            timesteps_metrics.append(timestep_metric)        
        
        return timesteps_metrics
