import os
from typing import Tuple
import boto3
import pandas as pd
import numpy as np
from functools import lru_cache
import glob
import ast

from utils import load_config, get_logger


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

    def __init__(self, config_path: str):
        """Initialize dataset loader.

        Args:
            config_path: Path to YAML config file
        """
        self.config = load_config(config_path, "dataset_features", self.REQUIRED_FIELDS)
        self.data_dir = os.path.join("data")
        self.logger = get_logger(__name__)

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
                self.logger.warning(
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
                    self.logger.info(f"Loaded instance info for {region}")
                    instance_info_dfs.append(region_instance_info_df)

                # Load prices
                region_prices_df = self.read_prices_files(region)
                if not region_prices_df.empty:
                    self.logger.info(f"Loaded prices for {region}")
                    prices_dfs.append(region_prices_df)

            if not prices_dfs or not instance_info_dfs:
                self.logger.warning(
                    "No instances matched the specified filters. Debug information:"
                )

            # Combine data
            prices_df = pd.concat(prices_dfs, ignore_index=True)
            instance_info_df = pd.concat(instance_info_dfs)

            # Filter prices_df to only include instances from instance_info_df
            valid_instance_ids = instance_info_df.index.tolist()
            prices_df = prices_df[prices_df["id_instance"].isin(valid_instance_ids)]

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

            if "instance_filters" in self.config:
                filters = self.config.get("instance_filters", {})
                self.logger.info(
                    f"Filters being applied: {filters}"
                )  # Self.logger.info all filters

                if "instance_family" in filters and filters["instance_family"]:
                    self.logger.info(
                        f"Applying instance_family filter: {filters['instance_family']}"
                    )
                    instance_info_df = instance_info_df[
                        instance_info_df["instance_family"].isin(
                            filters["instance_family"]
                        )
                    ]
                    self.logger.info(
                        f"Instances after instance_family filter: {len(instance_info_df)}"
                    )

                if "architectures" in filters and filters["architectures"]:
                    self.logger.info(
                        f"Applying architectures filter: {filters['architectures']}"
                    )
                    instance_info_df = instance_info_df[
                        instance_info_df["architectures"].apply(
                            lambda x: any(
                                arch in filters["architectures"] for arch in x
                            )
                        )
                    ]
                    self.logger.info(
                        f"Instances after architectures filter: {len(instance_info_df)}"
                    )

                if "generation" in filters and filters["generation"]:
                    self.logger.info(
                        f"Applying generation filter: {filters['generation']}"
                    )
                    instance_info_df = instance_info_df[
                        instance_info_df["generation"]
                        .astype(str)
                        .isin([str(g) for g in filters["generation"]])
                    ]
                    self.logger.info(
                        f"Instances after generation filter: {len(instance_info_df)}"
                    )

                if "size" in filters and filters["size"]:
                    self.logger.info(f"Applying size filter: {filters['size']}")
                    instance_info_df = instance_info_df[
                        instance_info_df["size"].isin(filters["size"])
                    ]
                    self.logger.info(
                        f"Instances after size filter: {len(instance_info_df)}"
                    )

                if "product_description" in filters and filters["product_description"]:
                    self.logger.info(
                        f"Applying product_description filter: {filters['product_description']}"
                    )
                    product_descriptions = filters["product_description"]
                    instance_info_df = instance_info_df[
                        instance_info_df["product_description"].isin(
                            product_descriptions
                        )
                    ]
                    self.logger.info(
                        f"Instances after product_description filter: {len(instance_info_df)}"
                    )

                if "instance_type" in filters and filters["instance_type"]:
                    self.logger.info(
                        f"Applying instance_type filter: {filters['instance_type']}"
                    )
                    instance_info_df = instance_info_df[
                        instance_info_df["instance_type"].isin(filters["instance_type"])
                    ]
                    self.logger.info(
                        f"Instances after instance_type filter: {len(instance_info_df)}"
                    )

                if "metal" in filters and not filters["metal"]:
                    self.logger.info(f"Applying metal filter: {filters['metal']}")
                    instance_info_df = instance_info_df[
                        ~instance_info_df["size"].str.contains(
                            "metal", case=False, na=False
                        )
                    ]
                    self.logger.info(
                        f"Instances after metal filter: {len(instance_info_df)}"
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
        """Download required files from S3.

        Uses an incremental download strategy to only fetch new or updated files
        when the data directory already exists, reducing bandwidth and time.
        Instance info files are always downloaded as they're critical for the dataset.
        """
        try:
            bucket_name = "spot-datasets"
            regions = self.config["regions"]

            if not os.path.exists(self.data_dir):
                self.logger.warning(
                    f"Data directory {self.data_dir} does not exist. Creating it and downloading all files..."
                )
                os.makedirs(self.data_dir)
            else:
                self.logger.info("Data directory exists, checking for latest files...")

            earliest_date = self._find_earliest_date_across_regions(regions)
            self._download_files_from_s3(bucket_name, regions, earliest_date)

        except Exception as e:
            self.logger.error(f"Failed to download files: {str(e)}")
            raise

    def _find_earliest_date_across_regions(self, regions):
        """Find the minimum end date among the latest files across all regions.

        This identifies the oldest data we need to update, ensuring complete
        time series data across all regions.

        Returns:
            pd.Timestamp or None: The earliest date found, or None if no valid files exist
        """
        earliest_date = None

        for region in regions:
            price_files = glob.glob(f"{self.data_dir}/prices_{region}_*.csv")
            if not price_files:
                # If any region has no files, we need complete data from start date
                earliest_date = self.config["start_date"]
                continue

            try:
                # Find the most recent file (by end date in filename)
                latest_file = sorted(price_files, key=lambda x: x.split("_")[-1])[-1]
                file_name = os.path.basename(latest_file)

                parts = file_name.split("_")
                if len(parts) >= 4:
                    end_date_str = parts[-1].split(".")[0]
                    end_date = pd.Timestamp(end_date_str).date()

                    # Track the minimum end date - we need all files from this date onward
                    if earliest_date is None or end_date < earliest_date:
                        earliest_date = end_date
                        self.logger.info(
                            f"Latest file for {region}: {file_name}, end date: {end_date}"
                        )
            except Exception as e:
                self.logger.error(f"Error processing files for {region}: {e}")

        return earliest_date

    def _download_files_from_s3(self, bucket_name, regions, earliest_date):
        """Download files from S3 with region and date filtering.

        Prioritizes instance_info files which contain critical metadata,
        then selectively downloads price files based on date range.

        Args:
            bucket_name: S3 bucket name
            regions: List of AWS regions to download files for
            earliest_date: Earliest date to download files from, or None for all files
        """
        if earliest_date:
            self.logger.info(
                f"Downloading files from {earliest_date} onwards and all instance info files..."
            )
        else:
            self.logger.info("Downloading all files for specified regions...")

        # Track regions with instance_info files to detect missing ones
        instance_info_downloaded = set()

        paginator = self.s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # Extract region from filename if it exists
                region_match = next(
                    (
                        region
                        for region in regions
                        if f"_{region}" in key or f"_{region}." in key
                    ),
                    None,
                )

                if not region_match:
                    continue

                # Prioritize instance_info files
                if key.startswith("instance_info_"):
                    if region_match not in instance_info_downloaded:
                        self._download_single_file(bucket_name, key)
                        instance_info_downloaded.add(region_match)
                elif self._should_download_file(key, earliest_date):
                    # Download price files based on date criteria
                    self._download_single_file(bucket_name, key)

        # Alert if any regions are missing instance info
        missing_regions = set(regions) - instance_info_downloaded
        if missing_regions:
            self.logger.warning(
                f"Could not find instance_info files for regions: {missing_regions}"
            )

    def _should_download_file(self, key, earliest_date):
        """Determine if a file should be downloaded based on our date criteria.

        Instance info files are always downloaded. For price files, we only download
        those with end dates after our earliest known date to avoid redundant downloads.

        Args:
            key: S3 object key
            earliest_date: Earliest date to consider for download

        Returns:
            bool: True if file should be downloaded
        """
        # No date filter means download everything
        if earliest_date is None:
            return True

        # Skip date filtering for non-price files
        if not key.startswith("prices_"):
            return True

        # Extract and check end date for price files
        parts = key.split("_")
        if len(parts) >= 4:
            try:
                end_date_str = parts[-1].split(".")[0]
                end_date = pd.Timestamp(end_date_str).date()
                # Only download files with data in our target range
                return end_date >= earliest_date
            except Exception:
                # Download file if we can't parse date (fail safe)
                return True

        return True

    def _download_single_file(self, bucket_name, key):
        """Download a file from S3, creating directories as needed."""
        local_path = os.path.join(self.data_dir, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(bucket_name, key, local_path)
        self.logger.info(f"Downloaded {key}")

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
