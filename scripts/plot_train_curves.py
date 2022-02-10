import argparse
import os

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')

    fig, ax = plt.subplots()
    for model in args.inputs:
        path = os.path.join(model, 'losses.npz')
        data = np.load(path)
        l, = ax.plot(data['train'], label=model)
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
    parser.add_argument('--ymin', type=float)
    parser.add_argument('--ymax', type=float)
    args = parser.parse_args()
    main()
