import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import numpy as np

from brever.modelmanagement import get_feature_indices


def main(inputs, features, label, bins, alpha):
    for feature in features:
        fig, ax = plt.subplots()
        for dataset_dir in inputs:
            feature_indices = get_feature_indices(dataset_dir, [feature])
            i_start, i_end = feature_indices[0]
            dataset_path = os.path.join(dataset_dir, 'dataset.hdf5')
            with h5py.File(dataset_path, 'r') as f:
                data = f['features'][:, i_start:i_end]
            ax.hist(data.flatten(), bins=bins, alpha=alpha)
            ax.set_title(feature)
        ax.legend(inputs)

    if label:
        fig, ax = plt.subplots()
        for dataset_dir in inputs:
            dataset_path = os.path.join(dataset_dir, 'dataset.hdf5')
            with h5py.File(dataset_path, 'r') as f:
                data = f['labels'][:]
            ax.hist(data.flatten(), bins=bins, alpha=alpha)
            ax.set_title('label')
        ax.legend(inputs)

        fig, ax = plt.subplots()
        n_inputs = len(inputs)
        offsets = np.arange(n_inputs)/(n_inputs+1)
        offsets = offsets - offsets.mean()
        bands = ['Low', 'Mid', 'High']
        n_bands = len(bands)
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        legend_handles = []
        for i, dataset_dir in enumerate(inputs):
            dataset_path = os.path.join(dataset_dir, 'dataset.hdf5')
            with h5py.File(dataset_path, 'r') as f:
                data = f['labels'][:]
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
            ax.set_title('label')
            legend_handle = mpatches.Patch(color=color, label=dataset_dir)
            legend_handles.append(legend_handle)
        ax.set_xticks(np.arange(n_bands))
        ax.set_xticklabels(bands)
        ax.set_xlabel('Frequency')
        ax.legend(handles=legend_handles, loc=9)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot feature distribution')
    parser.add_argument('input', nargs='+',
                        help='input dataset directories')
    parser.add_argument('--features', nargs='+', default=[],
                        help='features to plot')
    parser.add_argument('--label', action='store_true',
                        help='plot label')
    parser.add_argument('--bins', type=int, default=50,
                        help='number of bins')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='transparency')
    args = parser.parse_args()
    main(args.input, args.features, args.label, args.bins, args.alpha)
