import os
from glob import glob

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

import brever.modelmanagement as bmm


def main(models, args, **kwargs):
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')

    models = bmm.find_model(models=models, **kwargs)
    for model in models:
        print(model)

    fig, ax = plt.subplots()
    for model in models:
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
    parser = bmm.ModelFilterArgParser(description='plot training curves')
    parser.add_argument('input', nargs='+',
                        help='list of models whose curves to plot')
    parser.add_argument('--ymin', type=float)
    parser.add_argument('--ymax', type=float)
    filter_args, args = parser.parse_args()

    model_dirs = []
    for input_ in args.input:
        if not glob(input_):
            print(f'Model not found: {input_}')
        model_dirs += glob(input_)

    main(model_dirs, args, **vars(filter_args))
