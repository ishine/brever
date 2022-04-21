import argparse
import os

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from brever.args import ModelArgParser
from brever.config import get_config


def main():
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')

    fig, ax = plt.subplots()
    for model in args.inputs:
        path = os.path.join(model, 'losses.npz')
        data = np.load(path)

        if args.legend_params is None:
            label = model
        else:
            config = get_config(os.path.join(model, 'config.yaml'))
            label = {}
            for param_name in args.legend_params:
                label[param_name] = config.get_field(
                    ModelArgParser.arg_map[config.ARCH][param_name]
                )
            label = [f'{key}: {val}' for key, val in label.items()]
            label = ', '.join(label)

        l, = ax.plot(data['train'], label=label)
        _, = ax.plot(data['val'], '--', color=l.get_color())

    lines = [
        Line2D([], [], color='k', linestyle='-'),
        Line2D([], [], color='k', linestyle='--'),
    ]

    lh = ax.legend(loc=1)
    ax.legend(lines, ['train', 'val'], loc=2)
    ax.add_artist(lh)

    ax.set_ylim(args.ymin, args.ymax)

    ax.grid()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot training curves')
    parser.add_argument('inputs', nargs='+',
                        help='paths to model directories')
    parser.add_argument('--legend-params', nargs='+',
                        help='hyperparameters to use to label curves')
    parser.add_argument('--ymin', type=float)
    parser.add_argument('--ymax', type=float)
    args = parser.parse_args()
    main()
