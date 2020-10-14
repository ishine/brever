import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import yaml
import scipy.io

from brever.modelmanagement import (get_dict_field, ModelFilterArgParser,
                                    find_model, arg_to_keys_map)


def check_models(models, dims):
    values = []
    models_ = []
    for model in models:
        pesq_file = os.path.join('models', model, 'pesq_scores.mat')
        mse_file = os.path.join('models', model, 'mse_scores.npy')
        config_file = os.path.join('models', model, 'config_full.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if not (os.path.exists(pesq_file) and os.path.exists(mse_file)):
            print(f'Model {model} is not evaluated!')
            continue
        val = {dim: get_dict_field(config, arg_to_keys_map[dim])
               for dim in dims}
        if val not in values:
            values.append(val)
            models_.append(model)
        else:
            raise ValueError(f'Found more than one model for value {val}')
    return models_, values


def group_by_dimension(models, values, dimension):
    # first make groups
    if dimension is None:
        groups = []
        group_outer_values = []
        for model, val in zip(models, values):
            groups.append([{'model': model, 'val': val}])
            if val not in group_outer_values:
                group_outer_values.append(val)
    else:
        group_outer_values = []
        groups = []
        group_inner_values = []
        for model, val in zip(models, values):
            group_outer_val = {dimension: val[dimension]}
            if group_outer_val not in group_outer_values:
                group_outer_values.append(group_outer_val)
                groups.append([{'model': model, 'val': val}])
            else:
                index = group_outer_values.index(group_outer_val)
                groups[index].append({'model': model, 'val': val})
            group_inner_val = val.copy()
            group_inner_val.pop(dimension)
            if group_inner_val not in group_inner_values:
                group_inner_values.append(group_inner_val)
    if dimension is None:
        return groups, group_outer_values
    # then match order across groups
    # first sort the list of values
    for dim in group_inner_values[0].keys():
        group_inner_values = sorted(group_inner_values, key=lambda x: x[dim])
    # then make all groups have that order
    for i, group in enumerate(groups):
        group_sorted = []
        group_inner_vals_local = [model['val'].copy() for model in group]
        for val in group_inner_vals_local:
            val.pop(dimension)
        for group_inner_val in group_inner_values:
            if group_inner_val in group_inner_vals_local:
                index = group_inner_vals_local.index(group_inner_val)
                group_sorted.append(group[index])
        groups[i] = group_sorted
    return groups, group_outer_values


def load_pesq_and_mse(groups):
    for group in groups:
        for i in range(len(group)):
            model = group[i]['model']
            pesq_filepath = os.path.join('models', model, 'pesq_scores.mat')
            pesq = scipy.io.loadmat(pesq_filepath)['scores']
            mse_filepath = os.path.join('models', model, 'mse_scores.npy')
            mse = np.load(mse_filepath)
            group[i]['pesq'] = pesq
            group[i]['mse'] = mse


def sort_groups_by_mean_pesq(groups):
    groups_mean_pesq = []
    for group in groups:
        mean_pesqs = [model['pesq'].mean() for model in group]
        group_mean_pesq = np.mean(mean_pesqs)
        groups_mean_pesq.append(group_mean_pesq)
    indexes = np.argsort(groups_mean_pesq)
    groups = [groups[i] for i in indexes]
    return groups


def paths_to_dirnames(paths):
    dirnames = []
    for path in paths:
        dirnames.append(os.path.basename(os.path.normpath(path)))
    return dirnames


class LegendFormatter:
    def __init__(self, figure):
        self.figure = figure
        self.lh = figure.legend(loc=9)
        self.showed = False
        self.figure.canvas.mpl_connect('draw_event', self)
        self.figure.canvas.mpl_connect('resize_event', self)

    def __call__(self, event):
        if self.figure._cachedRenderer is None:
            return
        if event.name == 'draw_event' and self.showed:
            return
        self.showed = True
        lbbox = self.lh.get_window_extent()
        fig_width = self.figure.get_figwidth()*self.figure.dpi
        ncol = int(fig_width//(lbbox.width/self.lh._ncol))
        ncol = ncol//2+1
        if ncol != self.lh._ncol:
            self.lh.remove()
            self.lh = self.figure.legend(loc=9, ncol=ncol)
            self(event)
        else:
            fig_height = self.figure.get_figheight()*self.figure.dpi
            ratio = 1 - (lbbox.height + 20)/fig_height
            self.figure.tight_layout()
            try:
                self.figure.subplots_adjust(top=ratio)
            except ValueError:
                pass


def main(models, dimensions, group_by, no_sort, filter_):
    models = paths_to_dirnames(models)
    possible_models = find_model(**filter_)
    models = [model for model in models if model in possible_models]

    if group_by is not None and group_by not in dimensions:
        dimensions.append(group_by)

    models, values = check_models(models, dimensions)
    groups, group_values = group_by_dimension(models, values, group_by)
    load_pesq_and_mse(groups)
    if not no_sort:
        groups = sort_groups_by_mean_pesq(groups)
    else:
        for dim in group_values[0].keys():
            group_vals_sorted = sorted(group_values, key=lambda x: x[dim])
            i_sorted = [group_values.index(val) for val in group_vals_sorted]
            groups = [groups[i] for i in i_sorted]

    snrs = [0, 3, 6, 9, 12, 15]
    room_names = ['A', 'B', 'C', 'D']

    n = len(models)
    width = 1/(n+1)

    for ylabel, metric in zip(
                ['MSE', r'$\Delta PESQ$'],
                ['mse', 'pesq'],
            ):
        fig, axes = plt.subplots(1, 2, sharey=True)
        for axis, (ax, xticklabels, xlabel) in enumerate(zip(
                    axes[::-1],
                    [room_names, snrs],
                    ['Room', 'SNR (dB)'],
                )):
            model_count = 0
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            hatch_cycle = ['', '////', '\\\\\\', 'xxxx']
            for i, group in enumerate(groups):
                for j, model in enumerate(group):
                    data = model[metric]
                    if metric == 'mse':
                        mean = data.mean(axis=axis)
                        mean = np.hstack((mean, data.mean()))
                        err = None
                    elif metric == 'pesq':
                        mean = data.mean(axis=(axis, -1))
                        mean = np.hstack((mean, data.mean()))
                        err = data.std(axis=(axis, -1))
                        err = err/(data.shape[axis]*data.shape[-1])**0.5
                        err = np.hstack((err, data.std()))
                        err[-1] = err[-1]/(data.size)**0.5
                    if axis == 0:
                        label = f'{model["val"]}'
                    else:
                        label = None
                    x = np.arange(len(mean)) + (model_count - (n-1)/2)*width
                    x[-1] = x[-1] + 2*width
                    ax.bar(
                        x=x,
                        height=mean,
                        width=width,
                        label=label,
                        color=color_cycle[i % len(color_cycle)],
                        hatch=hatch_cycle[j % len(hatch_cycle)],
                        yerr=err,
                    )
                    model_count += 1
            xticks = np.arange(len(xticklabels) + 1, dtype=float)
            xticks[-1] = xticks[-1] + 2*width
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels + ['Mean'])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.yaxis.set_tick_params(labelleft=True)

        LegendFormatter(fig)

    plt.show()


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='compare models')
    parser.add_argument('input', nargs='+',
                        help='list of models to compare')
    parser.add_argument('--dims', nargs='+', required=True,
                        type=lambda x: x.replace('-', '_'),
                        help='parameter dimensions to compare')
    parser.add_argument('--group-by',
                        type=lambda x: x.replace('-', '_'),
                        help='parameter dimension to group by')
    parser.add_argument('--no-sort', action='store_true',
                        help='disable sorting by mean score')
    filter_args, args = parser.parse_args()

    if len(args.input) == 1:
        args.input = glob(args.input[0])
    main(args.input, args.dims, args.group_by, args.no_sort, vars(filter_args))
