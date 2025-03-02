import os
import logging
from typing import Tuple
import boto3
import pandas as pd
import numpy as np
from functools import lru_cache
import glob
import ast

from utils import load_config


class LoadSpotDataset:
    """Dataset loader for spot price prediction training.

    Handles downloading, preprocessing and loading spot price datasets from S3
    and instance metadata for machine learning training.

    Attributes:
        config: Dict containing configuration parameters
        data_dir: Directory to store downloaded data
        logger: Logger instance for debug/error messages
    """

    REQUIRED_FIELDS = [
        "regions",
        "data_folder",
        "time_col",
        "target_col",
        "timestep_hours",
    ]

    def __init__(self, config_path: str = "config.yaml", data_dir: str = "data"):
        """Initialize dataset loader.

        Args:
            config_path: Path to YAML config file
        """
        self.config = load_config(config_path, "dataset_features", self.REQUIRED_FIELDS)
        self.data_dir = os.path.join(data_dir)
        self.logger = logging.getLogger(__name__)

        try:
            self.s3 = boto3.client("s3")
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise

    @lru_cache(maxsize=32)
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load price and instance data with caching.

        Returns:
            Tuple of (prices_df, instance_info_df)

        Raises:
            Exception if data loading fails
        """
        try:
            if not os.path.exists(self.data_dir):
                self.logger.info(
                    f"{self.data_dir} does not exist. Downloading files from S3..."
                )
                self._download_files()

            prices_dfs = []
            instance_info_dfs = []

            for region in self.config["regions"]:
                self.logger.info(f"Loading info from {region}")
                # Load instance info
                region_instance_info_df = self.read_instance_info(region)
                if not region_instance_info_df.empty:
                    instance_info_dfs.append(region_instance_info_df)

                # Load prices
                region_prices_df = self.read_prices_files(region)
                if not region_prices_df.empty:
                    prices_dfs.append(region_prices_df)

            if not prices_dfs or not instance_info_dfs:
                raise ValueError("No data loaded for any region")

            # Combine data
            prices_df = pd.concat(prices_dfs, ignore_index=True)
            instance_info_df = pd.concat(instance_info_dfs)

            # Validate
            self._validate_loaded_data(prices_df, instance_info_df)

            return prices_df, instance_info_df

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def get_training_validation_test_split(
        self, prices_df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training, validation and test sets."""
        try:
            # Sort by time
            prices_df = prices_df.sort_values(self.config["time_col"])

            # Calculate split indices
            train_idx = int(len(prices_df) * train_ratio)
            val_idx = int(len(prices_df) * (train_ratio + val_ratio))

            # Split data
            train_df = prices_df.iloc[:train_idx]
            val_df = prices_df.iloc[train_idx:val_idx]
            test_df = prices_df.iloc[val_idx:]

            return train_df, val_df, test_df

        except Exception as e:
            self.logger.error(f"Failed to split data: {str(e)}")
            raise

    def read_instance_info(self, region: str) -> pd.DataFrame:
        """Read and preprocess instance metadata for a region."""
        try:
            instance_info_file = (
                f"{self.config['data_folder']}/instance_info_{region}.csv"
            )
            instance_info_df = pd.read_csv(instance_info_file)

            instance_info_df["id_instance"] = instance_info_df["id"]
            instance_info_df = instance_info_df.drop("id", axis=1).set_index(
                "id_instance"
            )

            instance_info_df["memory"] = instance_info_df["memory"].astype(int)
            instance_info_df["modifiers"] = (
                instance_info_df["modifiers"].fillna("").apply(lambda x: list(x))
            )
            instance_info_df["architectures"] = instance_info_df["architectures"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

            return instance_info_df
        except Exception as e:
            self.logger.error(f"Failed to read instance info for {region}: {str(e)}")
            return pd.DataFrame()

    def read_prices_files(self, region: str) -> pd.DataFrame:
        """Read and preprocess price data for a region."""
        try:
            # load prices on dataframes
            prices_files = glob.glob(f"{self.data_dir}/prices_{region}_*.csv")
            prices_df_list = [
                pd.read_csv(file)
                for file in prices_files
                if not pd.read_csv(file).empty
            ]
            if len(prices_df_list) == 0:
                raise Exception(f"No prices files for {region}")

            prices_df = pd.concat(prices_df_list, ignore_index=True)

            # modifications & validations on dataframe
            prices_df[self.config["time_col"]] = pd.to_datetime(
                prices_df[self.config["time_col"]], utc=True
            )
            prices_df = prices_df[
                (prices_df[self.config["time_col"]] >= self.config["start_date"])
                & (prices_df[self.config["time_col"]] <= self.config["end_date"])
            ]

            prices_df = self._group_prices_hour(prices_df)
            # TODO: ARREGLAR METODO
            # prices_df = self._add_time_features(prices_df, self.config["time_col"])

            return prices_df
        except Exception as e:
            self.logger.error(f"Failed to read prices for {region}: {str(e)}")
            return pd.DataFrame()

    def _download_files(self) -> None:
        """Download required files from S3."""
        try:
            os.makedirs(self.data_dir)
            bucket_name = "spot-datasets"
            regions = self.config["regions"]

            def download_file(key):
                local_path = os.path.join(self.data_dir, key)
                if not os.path.exists(os.path.dirname(local_path)):
                    os.makedirs(os.path.dirname(local_path))
                self.s3.download_file(bucket_name, key, local_path)
                print(f"Downloaded {key}")

            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name):
                for obj in page.get("Contents", []):
                    if any(f"_{region}" in obj["Key"] for region in regions):
                        download_file(obj["Key"])
        except Exception as e:
            self.logger.error(f"Failed to download files: {str(e)}")
            raise

    def _validate_loaded_data(
        self, prices_df: pd.DataFrame, instance_info_df: pd.DataFrame
    ) -> None:
        """Validate loaded dataframes."""
        if prices_df.empty:
            raise ValueError("Prices dataframe is empty")
        if instance_info_df.empty:
            raise ValueError("Instance info dataframe is empty")
        if prices_df[self.config["time_col"]].isna().any():
            raise ValueError(f"Missing values in {self.config['time_col']}")

    def _group_prices_hour(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        id_instances = prices_df["id_instance"].unique()
        complete_time_df = self._generate_complete_time_df(id_instances)

        prices_df[self.config["time_col"]] = prices_df[
            self.config["time_col"]
        ].dt.floor(f"{self.config['timestep_hours']}h")

        complete_time_df[self.config["time_col"]] = pd.to_datetime(
            complete_time_df[self.config["time_col"]], utc=True
        )

        prices_df = pd.merge(
            complete_time_df,
            prices_df,
            on=[self.config["time_col"], "id_instance"],
            how="left",
        )
        prices_df[self.config["target_col"]] = prices_df.groupby("id_instance")[
            self.config["target_col"]
        ].ffill()
        prices_df = prices_df.dropna(subset=[self.config["target_col"]])

        return prices_df

    def _add_time_features(self, df: pd.DataFrame, time_col: str, hours_in_day=24):
        def create_cyclical_features(values, period):
            """Convert cyclical features to sin/cos components"""
            values = values * 2 * np.pi / period
            return np.cos(values), np.sin(values)

        time_features = {
            "hour": (df[time_col].dt.hour, hours_in_day),
            "dayofweek": (df[time_col].dt.dayofweek, 7),
            "dayofmonth": (df[time_col].dt.day, df[time_col].dt.daysinmonth),
            "dayofyear": (df[time_col].dt.dayofyear, 365),
        }

        for feature, (values, period) in time_features.items():
            if feature in self.config.get("time_features", []):
                df[f"{feature}_cos"], df[f"{feature}_sin"] = create_cyclical_features(
                    values, period
                )
        return df

    def _generate_complete_time_df(self, id_instances: np.ndarray) -> pd.DataFrame:
        complete_time_range = pd.date_range(
            start=self.config["start_date"],
            end=self.config["end_date"],
            freq=f"{self.config['timestep_hours']}h",
        )
        complete_time_df = (
            pd.DataFrame({self.config["time_col"]: complete_time_range})
            .assign(key=1)
            .merge(pd.DataFrame({"id_instance": id_instances, "key": 1}), on="key")
            .drop("key", axis=1)
        )
        return complete_time_df

    def _clear_cache(self):
        """Clear cached data."""
        self.load_data.cache_clear()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self._clear_cache()
