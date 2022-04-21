import argparse
import os

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch

from brever.args import ModelArgParser
from brever.config import get_config


def pretty_table(dict_: dict, key_header: str = '') -> None:
    if not dict_:
        raise ValueError('input is empty')
    keys = dict_.keys()
    values = dict_.values()
    first_col_width = max(max(len(str(key)) for key in keys), len(key_header))
    col_widths = [first_col_width]
    for i, value in enumerate(values):
        if i == 0:
            sub_keys = value.keys()
        elif value.keys() != sub_keys:
            raise ValueError('values in input do not all have same keys')
    for key in sub_keys:
        col_width = max(max(len(str(v[key])) for v in values), len(key))
        col_widths.append(col_width)
    row_fmt = ' '.join(f'{{:<{width}}} ' for width in col_widths)
    print(row_fmt.format(key_header, *sub_keys))
    print(row_fmt.format(*['-'*w for w in col_widths]))
    for key, items in dict_.items():
        print(row_fmt.format(key, *items.values()))


def fmt_time(time_):
    h, m, s = int(time_//3600), int((time_ % 3600)//60), int(time_ % 60)
    return f'{h:>2} h {m:>2} m {s:>2} s'


def fmt_memory(memory):
    memory = round(memory/1e9, 2)
    return f'{memory:>5} GB'


def main():
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')

    summary = {}

    fig, ax = plt.subplots()
    for model in args.inputs:
        path = os.path.join(model, 'losses.npz')
        data = np.load(path)

        summary[model] = {}

        if args.legend_params is None:
            label = model
        else:
            config = get_config(os.path.join(model, 'config.yaml'))
            label = {}
            for param_name in args.legend_params:
                param_val = config.get_field(
                    ModelArgParser.arg_map[config.ARCH][param_name]
                )
                label[param_name] = param_val
                summary[model][param_name] = param_val
            label = [f'{key}: {val}' for key, val in label.items()]
            label = ', '.join(label)

        l, = ax.plot(data['train'], label=label)
        _, = ax.plot(data['val'], '--', color=l.get_color())

        state = torch.load(os.path.join(model, 'checkpoint.pt'))
        summary[model]['training time'] = fmt_time(state['time_spent'])
        summary[model]['GPU usage'] = fmt_memory(state['max_memory_allocated'])

    lines = [
        Line2D([], [], color='k', linestyle='-'),
        Line2D([], [], color='k', linestyle='--'),
    ]
    lh = ax.legend(loc=1)
    ax.legend(lines, ['train', 'val'], loc=2)
    ax.add_artist(lh)
    ax.set_ylim(args.ymin, args.ymax)
    ax.grid()

    pretty_table(summary)

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
