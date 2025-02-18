import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict

from utils import get_device, load_config
from .model import Model


class Evaluate:
    REQUIRED_FIELDS = [
        "eval_step",
        "prediction_length",
        "n_timesteps_metrics",
    ]

    def __init__(
        self, model: Model, config_path: str = "config.yaml", output_dir: str = "output"
    ):
        self.model = model
        self.device = get_device()
        self.config = load_config(config_path, "evaluate_config", self.REQUIRED_FIELDS)
        self.dataset_config = load_config(
            config_path, "dataset_config", ["sequence_length"]
        )
        os.makedirs(output_dir, exist_ok=True)

        self.segmented_metrics: DefaultDict[int, List[Dict]] = defaultdict(list)
        self.prediction_results: DefaultDict[
            int, List[Tuple[np.ndarray, np.ndarray]]
        ] = defaultdict(list)
        self.failed_instances: List[int] = []

    @property
    def metrics(self):
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
        return self.prediction_results[id_instance]

    def evaluate_instance(
        self, instance_data: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        required_len = (
            self.dataset_config["sequence_length"] + self.config["prediction_length"]
        )

        if len(instance_data) < required_len:
            return []

        num_experiments = (len(instance_data) - required_len) // self.config[
            "eval_step"
        ] + 1

        input_sequences = np.zeros(
            (num_experiments, self.dataset_config["sequence_length"])
        )
        target_sequences = np.zeros((num_experiments, self.config["prediction_length"]))

        for step in range(num_experiments):
            start_idx = step * self.config["eval_step"]
            end_idx = start_idx + required_len

            if end_idx > len(instance_data):
                num_experiments = step
                input_sequences = input_sequences[:step]
                target_sequences = target_sequences[:step]
                break

            input_sequences[step] = instance_data[
                start_idx : start_idx + self.dataset_config["sequence_length"]
            ]
            target_sequences[step] = instance_data[
                start_idx + self.dataset_config["sequence_length"] : end_idx
            ]

        if num_experiments == 0:
            return []

        with torch.no_grad():
            predictions = (
                self.model.forecast(
                    torch.FloatTensor(input_sequences).to(self.device),
                    self.config["prediction_length"],
                )
                .cpu()
                .numpy()
            )

        return list(zip(target_sequences, predictions))

    def evaluate_all(self, df: pd.DataFrame) -> Dict:
        print("Evaluation Configuration:")
        print(f"- Sequence length: {self.dataset_config['sequence_length']}")
        print(f"- Prediction length: {self.config['prediction_length']}")
        print(f"- Total instances: {df['id_instance'].nunique()}\n")

        instance_groups = {
            id_: group["spot_price"].values for id_, group in df.groupby("id_instance")
        }

        with tqdm(total=len(instance_groups), desc="Processing instances") as pbar:
            for instance_id, instance_data in instance_groups.items():
                results = self.evaluate_instance(np.array(instance_data))

                if results:
                    self.prediction_results[instance_id] = results  # type: ignore
                    list_target = [r[0] for r in results]
                    list_pred = [r[1] for r in results]
                    self.segmented_metrics[instance_id] = self._calculate_metrics(  # type: ignore
                        list_pred, list_target, self.config["n_timesteps_metrics"]
                    )
                else:
                    self.failed_instances.append(instance_id)  # type: ignore

                pbar.update(1)

        if self.failed_instances:
            print(
                f"\nWarning: {len(self.failed_instances)} instances had insufficient data for evaluation"
            )

        return self.segmented_metrics

    @staticmethod
    def _calculate_metrics(
        predictions: List[np.ndarray], targets: List[np.ndarray], n_timesteps: int
    ) -> List[Dict]:
        np_pred = np.array(predictions)
        np_targets = np.array(targets)

        n_segments = np_pred.shape[1] // n_timesteps
        metrics = []

        for i in range(n_segments):
            start = i * n_timesteps
            end = start + n_timesteps

            pred_segments = np_pred[:, start:end]
            target_segments = np_targets[:, start:end]

            abs_diff = np.abs(pred_segments - target_segments)
            abs_sum = np.abs(pred_segments) + np.abs(target_segments)
            smape_values = 2 * abs_diff / abs_sum

            pred_diff = np.diff(pred_segments, axis=1)
            target_diff = np.diff(target_segments, axis=1)
            direction_match = np.sign(pred_diff) == np.sign(target_diff)

            metrics.append(
                {
                    "n_timestep": start,
                    "mape": float(np.mean(abs_diff / target_segments) * 100),
                    "smape": float(np.mean(smape_values) * 100),
                    "smape_std": float(np.std(smape_values) * 100),
                    "smape_cv": float(
                        np.std(smape_values) / np.mean(smape_values) * 100
                    ),
                    "rmse": float(
                        np.sqrt(np.mean((pred_segments - target_segments) ** 2))
                    ),
                    "direction_accuracy": float(np.mean(direction_match) * 100),
                }
            )

        return metrics
