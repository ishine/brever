import argparse
from glob import glob
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from brever.data import H5Dataset
import brever.management as bm


def get_data(dataset_dir, name, kind, long_term=False):
    if isinstance(dataset_dir, list):
        data = [get_data(dir_, name, kind, long_term) for dir_ in dataset_dir]
        data = np.vstack(data)
    else:
        if kind == 'feature':
            h5dataset = H5Dataset(dataset_dir, features=[name], load=True)
            data, _ = h5dataset[:]
        elif kind == 'label':
            if long_term:
                metadata_path = os.path.join(dataset_dir, 'mixture_info.json')
                metadata = bm.read_json(metadata_path)
                data = [item['lt_labels'][name] for item in metadata]
                data = np.asarray(data)
            else:
                h5dataset = H5Dataset(dataset_dir, labels=[name], load=True)
                _, data = h5dataset[:]
        else:
            raise ValueError('kind must be feature or label')
    return data


def make_boxplot(ax, inputs, item, kind, labels):
    n_inputs = len(inputs)
    offsets = np.arange(n_inputs)/(n_inputs+1)
    offsets = offsets - offsets.mean()
    bands = ['Low', 'Mid', 'High']
    n_bands = len(bands)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_handles = []
    for i, dataset_dir in enumerate(inputs):
        data = get_data(dataset_dir, item, kind)
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
    ax.set_title(item)
    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(bands)
    ax.set_xlabel('Frequency')
    ax.legend(handles=legend_handles, loc=9)


def main(args):

    def hist(ax, data, bins=args.bins, alpha=args.alpha):
        ax.hist(data.flatten(), bins=bins, alpha=alpha, density=True)

    globed_inputs = [glob(item) for item in args.inputs]

    for items, kind in [(args.features, 'feature'), (args.labels, 'label')]:
        for item in items:
            fig, ax = plt.subplots()
            for dataset_dirs in globed_inputs:
                data = get_data(dataset_dirs, item, kind)
                hist(ax, data)
            ax.set_title(item)
            ax.legend(args.inputs)

            fig, ax = plt.subplots()
            make_boxplot(ax, globed_inputs, item, kind, args.inputs)

    for item in args.labels:
        fig, ax = plt.subplots()
        for dataset_dirs in globed_inputs:
            data = get_data(dataset_dirs, item, 'label', long_term=True)
            hist(ax, data, bins=np.linspace(0, 1, 21))
        ax.set_title(f'long term {item}')
        ax.legend(args.inputs)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot feature distribution')
    parser.add_argument('inputs', nargs='+',
                        help='input dataset directories')
    parser.add_argument('--features', nargs='+', default=[],
                        help='features to plot')
    parser.add_argument('--labels', nargs='+', default=[],
                        help='labels to plot')
    parser.add_argument('--bins', type=int, default=50,
                        help='number of bins')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='transparency')
    args = parser.parse_args()

    if not args.features and not args.labels:
        print('No items to plot were selected. Use --features or --labels '
              'to select items to plot.')

    main(args)
