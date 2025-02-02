import yaml
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime, timezone, date


def plot_series(
    time,
    series,
    format="-",
    roof=None,
    title=None,
    xlabel="Time",
    ylabel="Value",
    legend=None,
):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    if isinstance(series, tuple):
        for i, series_num in enumerate(series):
            sns.lineplot(
                x=time,
                y=series_num,
                label=legend[i] if legend else None,
                linestyle=format,
            )
    else:
        sns.lineplot(
            x=time,
            y=series,
            label=legend if legend else None,
            linestyle=format,
        )

    if roof is not None:
        plt.ylim(bottom=0, top=roof)
    else:
        plt.ylim(bottom=0)

    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
        plt.legend()

    plt.show()


def load_config(config_path: str, config_key: str, required_fields: list[str]) -> dict:
    """Loads and validates configuration from a YAML file.
    This function reads a YAML configuration file, processes date fields by converting
    string dates to datetime objects with UTC timezone, and validates required fields.

    Args:
        - config_path (str): Path to the YAML configuration file
        - config_key (str): Key to extract from the root of the YAML document
        - required_fields (list[str]): List of field names that must be present in the config

    Returns:
        dict: Processed configuration dictionary with date strings converted to datetime objects
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)[config_key]

    for key in config:
        if key.endswith("_date"):
            if isinstance(config[key], str):
                config[key] = datetime.strptime(config[key], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            elif isinstance(config[key], date):
                config[key] = datetime.combine(
                    config[key], datetime.min.time()
                ).replace(tzinfo=timezone.utc)

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    return config
