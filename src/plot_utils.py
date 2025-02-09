"""Plotting utilities, currently for the resource scarcity model."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Define custom colormaps
cmap_bmi = LinearSegmentedColormap.from_list('custom_bmi', [
    (0.0, 'red'),
    ((16 - 15) / (22 - 15), 'orange'),
    ((18.5 - 15) / (22 - 15), 'yellow'),
    (1.0, 'green')
])

cmap_kcal = LinearSegmentedColormap.from_list('custom_kcal', [
    (0.0, 'red'),
    ((1000 - 600) / (1540 - 600), 'orange'),
    ((1300 - 600) / (1540 - 600), 'yellow'),
    (1.0, 'green')
])

cmap_mortality = LinearSegmentedColormap.from_list('custom_mortality', [
    (0.0, 'green'),
    (0.5, 'yellow'),
    (1.0, 'red')
])


def plot_percentiles_heatmap(data_pivot, title, color_params, output_params):
    """
    Plot a heatmap based on the given data pivot and parameters.

    Parameters:
    data_pivot (pd.DataFrame): Dataframe with data to plot.
    title (str): Plot title.
    color_params (dict): Contains:
        'cmap': colormap for the heatmap,
        'vmin': minimum value for colormap,
        'vmax': maximum value for colormap.
    output_params (dict): Contains:
        'x_labels': x-axis labels,
        'filename': file name for saving the plot,
        'show': boolean flag; if True, displays the plot.
    """
    data_pivot = data_pivot.apply(pd.to_numeric, errors='coerce')
    if data_pivot.isnull().sum().sum() > 0:
        print('Warning: Missing values detected. Forward filling missing values.')
    data_pivot = data_pivot.ffill()

    y_labels_full = list(range(1, 101))
    y_labels_to_show = [str(i) if i % 10 == 0 else '' for i in y_labels_full]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data_pivot,
        xticklabels=output_params.get('x_labels'),
        yticklabels=y_labels_to_show,
        cmap=color_params.get('cmap'),
        vmin=color_params.get('vmin'),
        vmax=color_params.get('vmax')
    )
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('Percentile Group')
    plt.tight_layout()

    if output_params.get('show', False):
        plt.show()
    else:
        plt.savefig(output_params.get('filename'))
        plt.close()
