""" Plotting utilities, currently for the resource scarcity model."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



# Define custom colormaps
cmap_bmi = LinearSegmentedColormap.from_list('custom_bmi', [
    (0.0, 'red'),
    ((16 - 15)/(22 - 15), 'orange'),
    ((18.5 - 15)/(22 - 15), 'yellow'),
    (1.0, 'green')
])

cmap_kcal = LinearSegmentedColormap.from_list('custom_kcal', [
    (0.0, 'red'),
    ((1000 - 600)/(1540 - 600), 'orange'),
    ((1300 - 600)/(1540 - 600), 'yellow'),
    (1.0, 'green')
])

cmap_mortality = LinearSegmentedColormap.from_list('custom_mortality', [
    (0.0, 'green'),
    (0.5, 'yellow'),
    (1.0, 'red')
])


def plot_percentiles_heatmap(data_pivot, title, cmap, vmin, vmax, filename, x_labels, show=False):
    data_pivot = data_pivot.apply(pd.to_numeric, errors='coerce')

    # Notify if there are any missing values
    if data_pivot.isnull().sum().sum() > 0:
        print('Warning: Missing values detected. Forward filling missing values.')
    
    # Handle missing values
    data_pivot = data_pivot.ffill()

    # Create y-axis labels: full range for percentiles
    y_labels_full = list(range(1, 101))

    # Create y-axis labels to display: empty for non-10th percentiles
    y_labels_to_show = [str(i) if i % 10 == 0 else '' for i in y_labels_full]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data_pivot,
        xticklabels=x_labels,  # months
        yticklabels=y_labels_to_show,  # Show only every 10th y-axis label
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('Percentile Group')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


