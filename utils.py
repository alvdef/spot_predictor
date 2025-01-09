import seaborn as sns
import torch 
from matplotlib import pyplot as plt

sns.set_theme(style="whitegrid")

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
    

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device
