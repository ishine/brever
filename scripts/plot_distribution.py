import argparse
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from brever.pytorchtools import H5Dataset


def get_data(dataset_dir, feature):
    if isinstance(dataset_dir, list):
        data = [get_data(dir_, feature) for dir_ in dataset_dir]
        data = np.vstack(data)
    else:
        if feature == 'label':
            h5dataset = H5Dataset(dataset_dir, None, load=True)
            _, data = h5dataset[:]
        else:
            h5dataset = H5Dataset(dataset_dir, [feature], load=True)
            data, _ = h5dataset[:]
    return data


def make_boxplot(ax, inputs, feature, labels):
    n_inputs = len(inputs)
    offsets = np.arange(n_inputs)/(n_inputs+1)
    offsets = offsets - offsets.mean()
    bands = ['Low', 'Mid', 'High']
    n_bands = len(bands)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_handles = []
    for i, dataset_dir in enumerate(inputs):
        data = get_data(dataset_dir, feature)
        n_labels = data.shape[1]
        color = color_cycle[i % len(color_cycle)]
        for j, band in enumerate(bands):
            i_start, i_end = n_labels//n_bands*j, n_labels//n_bands*(j+1)
            bplot = ax.boxplot(
                data[:, i_start:i_end].flatten(),
                positions=[j + offsets[i]],
                patch_artist=True,
            )
            bplot['boxes'][0].set_facecolor(color)
            bplot['medians'][0].set_color('k')
        legend_handle = mpatches.Patch(color=color, label=labels[i])
        legend_handles.append(legend_handle)
    ax.set_title('label')
    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(bands)
    ax.set_xlabel('Frequency')
    ax.legend(handles=legend_handles, loc=9)


def main(inputs, features, bins, alpha):

    def hist(ax, data):
        ax.hist(data.flatten(), bins=bins, alpha=alpha, density=True)

    globed_inputs = [glob(item) for item in inputs]

    for feature in features:
        fig, ax = plt.subplots()
        for dataset_dirs in globed_inputs:
            data = get_data(dataset_dirs, feature)
            hist(ax, data)
        ax.set_title(feature)
        ax.legend(inputs)

        fig, ax = plt.subplots()
        make_boxplot(ax, globed_inputs, feature, inputs)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot feature distribution')
    parser.add_argument('inputs', nargs='+',
                        help='input dataset directories')
    parser.add_argument('--features', nargs='+', default=[],
                        help='features to plot')
    parser.add_argument('--bins', type=int, default=50,
                        help='number of bins')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='transparency')
    args = parser.parse_args()
    main(args.inputs, args.features, args.bins, args.alpha)
