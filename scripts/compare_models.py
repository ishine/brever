import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import yaml
import scipy.io
import re

from brever.modelmanagement import (get_config_field, ModelFilterArgParser,
                                    find_model)
from brever.config import defaults


def check_models(models, dims):
    if dims is None:
        return models, models
    values = []
    models_ = []
    configs_ = []
    for model in models:
        pesq_file = os.path.join('models', model, 'pesq_scores.mat')
        mse_file = os.path.join('models', model, 'mse_scores.npy')
        config_file = os.path.join('models', model, 'config_full.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if not (os.path.exists(pesq_file) and os.path.exists(mse_file)):
            print(f'Model {model} is not evaluated!')
            continue
        val = {dim: get_config_field(config, dim)
               for dim in dims}
        if val not in values:
            values.append(val)
            models_.append(model)
            configs_.append(config)
        else:
            duplicate_config = configs_[values.index(val)]
            duplicate_model = models_[values.index(val)]
            if duplicate_config == config:
                print((f'Models {model} and {duplicate_model} both have the '
                       f'following parameters: {val}. One model configuration '
                       'is a subset of the other, meaning both models are '
                       f'likely to be identical. Model {duplicate_model} will '
                       'be skipped.'))
            else:
                raise ValueError((f'Models {model} and {duplicate_model} both '
                                  f'have the following parameters: {val}. '
                                  'The rest of the parameters differ. '
                                  'Consider using the --default option to set '
                                  'the rest of the parameters to their '
                                  'default value.'))
    return models_, values


def group_by_dimension(models, values, dimensions):
    # first make groups
    if dimensions is None:
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
            group_outer_val = {}
            for dimension in dimensions:
                group_outer_val[dimension] = val[dimension]
            if group_outer_val not in group_outer_values:
                group_outer_values.append(group_outer_val)
                groups.append([{'model': model, 'val': val}])
            else:
                index = group_outer_values.index(group_outer_val)
                groups[index].append({'model': model, 'val': val})
            group_inner_val = val.copy()
            for dimension in dimensions:
                group_inner_val.pop(dimension)
            if group_inner_val not in group_inner_values:
                group_inner_values.append(group_inner_val)
    if dimensions is None:
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
            for dimension in dimensions:
                val.pop(dimension)
        for group_inner_val in group_inner_values:
            if group_inner_val in group_inner_vals_local:
                index = group_inner_vals_local.index(group_inner_val)
                group_sorted.append(group[index])
        groups[i] = group_sorted
    return groups, group_outer_values


def load_scores(groups):
    for group in groups:
        for i in range(len(group)):
            model = group[i]['model']
            pesq_filepath = os.path.join('models', model, 'pesq_scores.mat')
            pesq = scipy.io.loadmat(pesq_filepath)['scores']
            mse_filepath = os.path.join('models', model, 'mse_scores.npy')
            mse = np.load(mse_filepath)
            seg_filepath = os.path.join('models', model, 'seg_scores.npy')
            seg = np.load(seg_filepath)
            segSSNR = seg[:, :, :, 0]
            segBR = seg[:, :, :, 1]
            segNR = seg[:, :, :, 2]
            segRR = seg[:, :, :, 3]
            group[i]['pesq'] = pesq
            group[i]['mse'] = mse
            group[i]['segSSNR'] = segSSNR
            group[i]['segBR'] = segBR
            group[i]['segNR'] = segNR
            group[i]['segRR'] = segRR
            train_filepath = os.path.join('models', model, 'train_losses.npy')
            train_curve = np.load(train_filepath)
            val_filepath = os.path.join('models', model, 'val_losses.npy')
            val_curve = np.load(val_filepath)
            group[i]['train_curve'] = train_curve
            group[i]['val_curve'] = val_curve
            pesq_filepath = os.path.join('models', model, 'pesq_scores_oracle.mat')
            pesq = scipy.io.loadmat(pesq_filepath)['scores_oracle']
            seg_filepath = os.path.join('models', model, 'seg_scores_oracle.npy')
            seg = np.load(seg_filepath)
            segSSNR = seg[:, :, :, 0]
            segBR = seg[:, :, :, 1]
            segNR = seg[:, :, :, 2]
            segRR = seg[:, :, :, 3]
            group[i]['oracle'] = {}
            group[i]['oracle']['pesq'] = pesq
            group[i]['oracle']['mse'] = np.zeros(mse.shape)
            group[i]['oracle']['segSSNR'] = segSSNR
            group[i]['oracle']['segBR'] = segBR
            group[i]['oracle']['segNR'] = segNR
            group[i]['oracle']['segRR'] = segRR


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


def set_default_parameters(filter_, dimensions, group_by):
    default_config = defaults().to_dict()
    for key, value in filter_.items():
        if (value is None and (dimensions is None or key not in dimensions)
                and (group_by is None or key not in group_by)):
            new_value = [get_config_field(default_config, key)]
            filter_[key] = new_value


def merge_lists(dimensions, group_by):
    if group_by is not None:
        if dimensions is None:
            dimensions = group_by.copy()
        else:
            for dimension in group_by:
                if dimensions not in group_by:
                    dimensions.append(dimension)
    return dimensions


def get_snrs_and_rooms(models):
    snrss = []
    roomss = []
    for model in models:
        config = defaults()
        config_file = os.path.join('models', model, 'config.yaml')
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))
        basename = os.path.basename(config.POST.PATH.TEST)
        basename = os.path.basename(config.POST.PATH.TEST)
        dirname = os.path.dirname(config.POST.PATH.TEST)
        r = re.compile(fr'^{basename}_(snr-?\d{{1,2}})_(.*)$')
        dirs_ = [dir_ for dir_ in filter(r.match, os.listdir(dirname))]
        snrs = sorted(set(r.match(dir_).group(1) for dir_ in dirs_))
        rooms = sorted(set(r.match(dir_).group(2) for dir_ in dirs_))
        snrss.append(snrs)
        roomss.append(rooms)
    assert all(snrs == snrss[0] for snrs in snrss)
    assert all(rooms == roomss[0] for rooms in roomss)
    return snrss[0], roomss[0]


class LegendFormatter:
    def __init__(self, figure, lh=None, ncol=None):
        self.figure = figure
        self.input_lh = lh
        if self.input_lh is None:
            self.lh = figure.legend(loc=9)
        else:
            self.lh = self.input_lh
        self.figure.canvas.mpl_connect('draw_event', self)
        self.figure.canvas.mpl_connect('resize_event', self)
        self.ncol = ncol

    def __call__(self, event):
        if self.figure._cachedRenderer is None:
            return
        lbbox = self.lh.get_window_extent()
        fig_width = self.figure.get_figwidth()*self.figure.dpi
        ncol = int(fig_width//(lbbox.width/self.lh._ncol))
        ncol = min(ncol, len(self.lh.legendHandles))
        ncol = max(1, ncol)
        if self.ncol is not None:
            ncol = min(ncol, self.ncol)
        if ncol != self.lh._ncol:
            self.lh.remove()
            if self.input_lh is None:
                self.lh = self.figure.legend(loc=9, ncol=ncol)
            else:
                handles = self.input_lh.legendHandles
                labels = [text.get_text() for text in self.input_lh.texts]
                self.lh = self.figure.legend(handles, labels, loc=9, ncol=ncol)
            self(event)
        else:
            fig_height = self.figure.get_figheight()*self.figure.dpi
            ratio = 1 - (lbbox.height + 20)/fig_height
            self.figure.tight_layout()
            try:
                self.figure.subplots_adjust(top=ratio)
            except ValueError:
                pass


def main(models, dimensions, group_by, sort_by, filter_, legend, top, ncol,
         default, only_pesq, figsize, ymax, train_curve, oracle):
    if default:
        set_default_parameters(filter_, dimensions, group_by)

    models = paths_to_dirnames(models)
    possible_models = find_model(**filter_)
    models = [model for model in models if model in possible_models]

    dimensions = merge_lists(dimensions, group_by)

    models, values = check_models(models, dimensions)
    groups, group_values = group_by_dimension(models, values, group_by)
    load_scores(groups)
    if sort_by == 'pesq':
        groups = sort_groups_by_mean_pesq(groups)
    elif sort_by == 'dims':
        if dimensions is not None:
            group_values_copy = group_values.copy()
            for dim in reversed(list(group_values[0].keys())):
                group_vals_sorted = sorted(group_values_copy,
                                           key=lambda x: str(x[dim]))
                i_sorted = [group_values_copy.index(val)
                            for val in group_vals_sorted]
                groups = [groups[i] for i in i_sorted]
                group_values_copy = [group_values_copy[i] for i in i_sorted]
        else:
            raise ValueError("Can't sort by dims when no dim is provided")
    elif sort_by is not None:
        raise ValueError('Unrecognized sort-by argument, must be pesq or dims')

    if top is not None:
        groups = groups[-top:]

    for group in groups:
        for model in group:
            print(model['model'])

    snrs, rooms = get_snrs_and_rooms(models)

    n = sum(len(group) for group in groups)
    if oracle:
        n *= 2
    width = 1/(n+1)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hatch_cycle = ['', '////', '\\\\\\', 'xxxx']
    for ylabel, metric in zip(
            ['MSE', r'$\Delta PESQ$', 'segSSNR', 'segBR', 'segNR', 'segRR'],
            ['mse', 'pesq', 'segSSNR', 'segBR', 'segNR', 'segRR'],
            ):
        if only_pesq and metric != 'pesq':
            continue
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=figsize)
        for axis, (ax, xticklabels, xlabel) in enumerate(zip(
                    axes[::-1],
                    [rooms, snrs],
                    ['Room', 'SNR (dB)'],
                )):
            model_count = 0
            for i, group in enumerate(groups):
                color = color_cycle[i % len(color_cycle)]
                hatch_count = 0
                for j, model in enumerate(group):
                    datas = [model[metric]]
                    if oracle:
                        datas.append(model['oracle'][metric])
                    for k, data in enumerate(datas):
                        hatch = hatch_cycle[hatch_count % len(hatch_cycle)]
                        if metric == 'mse':
                            mean = data.mean(axis=axis)
                            mean = np.hstack((mean, data.mean()))
                            err = None
                        else:
                            mean = data.mean(axis=(axis, -1))
                            mean = np.hstack((mean, data.mean()))
                            err = data.std(axis=(axis, -1))
                            err = err/(data.shape[axis]*data.shape[-1])**0.5
                            err = np.hstack((err, data.std()))
                            err[-1] = err[-1]/(data.size)**0.5
                        if axis == 0:
                            if legend is None:
                                label = f'{model["val"]}'
                                if k == 1:
                                    label += ' - oracle'
                            else:
                                label = legend[model_count]
                        else:
                            label = None
                        x = np.arange(len(mean)) + (model_count - (n-1)/2)*width
                        x[-1] = x[-1] + 2*width
                        ax.bar(x=x, height=mean, width=width, label=label,
                               color=color, hatch=hatch, yerr=err)
                        model_count += 1
                        hatch_count += 1

            xticks = np.arange(len(xticklabels) + 1, dtype=float)
            xticks[-1] = xticks[-1] + 2*width
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels + ['Mean'])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.yaxis.set_tick_params(labelleft=True)
            if metric == 'pesq' and ymax is not None:
                ax.set_ylim(0, ymax)
        LegendFormatter(fig, ncol=ncol)

    if not only_pesq:
        symbols = ['o', 's', '^', 'v', '<', '>']
        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
        fig_legend_handles = []
        fig_legend_labels = []
        for axis, (ax, labels) in enumerate(zip(
                    axes[::-1],
                    [rooms, snrs],
                )):
            ax_legend_handles = []
            ax_legend_labels = []
            for i, group in enumerate(groups):
                color = color_cycle[i % len(color_cycle)]
                for j, model in enumerate(group):
                    x = model['segBR'].mean(axis=(axis, -1))
                    y = model['segSSNR'].mean(axis=(axis, -1))
                    line, = ax.plot(x, y, linestyle='--',
                                    color=color)
                    if axis == 0:
                        fig_legend_handles.append(line)
                        fig_legend_labels.append(f'{model["val"]}')
                    for k, (x_, y_) in enumerate(zip(x, y)):
                        ax.plot(x_, y_, marker=symbols[k], markersize=10,
                                linestyle='', color=color)
                        if i == j == 0:
                            dummy_line, = ax.plot([], [], marker=symbols[k],
                                                  markersize=10, linestyle='',
                                                  color='k')
                            ax_legend_handles.append(dummy_line)
                            if axis == 0:
                                label = f'room {labels[k]}'
                            else:
                                label = f'SNR = {labels[k]} dB'
                            ax_legend_labels.append(label)
            ax.legend(ax_legend_handles, ax_legend_labels)
            ax.set_xlabel('segBR (dB)')
            ax.set_ylabel('segSSNR (dB)')
        lh = fig.legend(fig_legend_handles, fig_legend_labels)
        LegendFormatter(fig, lh=lh)

    if train_curve:
        fig, ax = plt.subplots(1, 1)
        for i, group in enumerate(groups):
            color = color_cycle[i % len(color_cycle)]
            for j, model in enumerate(group):
                if legend is None:
                    label = f'{model["val"]}'
                else:
                    label = legend[model_count]
                ax.plot(model['train_curve'], label=label, color=color)
                ax.plot(model['val_curve'], '--', color=color)
        LegendFormatter(fig, ncol=ncol)

    plt.show()


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='compare models')
    parser.add_argument('input', nargs='+',
                        help='list of models to compare')
    parser.add_argument('--dims', nargs='+',
                        type=lambda x: x.replace('-', '_'),
                        help='parameter dimensions to compare')
    parser.add_argument('--group-by', nargs='+',
                        type=lambda x: x.replace('-', '_'),
                        help='parameter dimension to group by')
    parser.add_argument('--sort-by',
                        help='how to sort the models')
    parser.add_argument('--legend', nargs='+',
                        help='custom legend')
    parser.add_argument('--ncol', type=int,
                        help='number of legend columns')
    parser.add_argument('--top', type=int,
                        help='only plot top best models')
    parser.add_argument('--default', action='store_true',
                        help='use default parameters to filter models')
    parser.add_argument('--only-pesq', action='store_true',
                        help='plot only pesq scores')
    parser.add_argument('--figsize', nargs=2, type=int,
                        help='figure size')
    parser.add_argument('--ymax', type=float,
                        help='pesq y axis upper limits')
    parser.add_argument('--train-curve', action='store_true',
                        help='plot training curves')
    parser.add_argument('--oracle', action='store_true',
                        help='plot oracle scores')
    filter_args, args = parser.parse_args()

    model_dirs = []
    for input_ in args.input:
        if not glob(input_):
            print(f'Model not found: {input_}')
        model_dirs += glob(input_)
    main(model_dirs, args.dims, args.group_by, args.sort_by, vars(filter_args),
         args.legend, args.top, args.ncol, args.default, args.only_pesq,
         args.figsize, args.ymax, args.train_curve, args.oracle)
