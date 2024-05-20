import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
from typing import *

def smooth_curve(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def plot_metrics(metrics_list: List[Dict[str, List[Tuple[int, Any]]]], names: List[str]):
    """
    Plots the metrics from different trainings on separate axes.

    Parameters:
    - metrics_list (List[Dict[str, List[Tuple[int, Any]]]]): A list where each element is a dictionary returned by CSVReader.read_metrics() method for different trainings.
    - names (List[str]): A list of names for each training to be used as labels in the plot.
    """
    # Determine the number of unique metrics across all trainings
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    # Create a subplot for each metric
    fig, axes = plt.subplots(nrows=len(all_metrics), ncols=1, figsize=(10, 5 * len(all_metrics)))
    if len(all_metrics) == 1:
        axes = [axes]  # Make it iterable
    
    for ax, metric in zip(axes, all_metrics):
        for name, metrics in zip(names, metrics_list):
            if metric in metrics:
                steps, values = zip(*metrics[metric])  # Unpack steps and values
                ax.plot(steps, values, label=name)
        ax.set_title(metric)
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.legend()

    plt.tight_layout()
    plt.show()