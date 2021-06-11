import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import yaml
import scipy.io
import math

from brever.modelmanagement import (get_config_field, ModelFilterArgParser,
                                    find_model)
from brever.config import defaults


def check_models(models, dims):
    models_dir = defaults().PATH.MODELS
    if dims is None:
        models_ = []
        for model in models:
            mat_file = os.path.join(models_dir, model, 'scores.mat')
            npz_file = os.path.join(models_dir, model, 'scores.npz')
            if not os.path.exists(mat_file) or not os.path.exists(npz_file):
                print(f'Model {model} is not evaluated!')
                continue
            models_.append(model)
        return models_, models_
    values = []
    models_ = []
    configs_ = []
    for model in models:
        mat_file = os.path.join(models_dir, model, 'scores.mat')
        npz_file = os.path.join(models_dir, model, 'scores.npz')
        config_file = os.path.join(models_dir, model, 'config_full.yaml')
        if not os.path.exists(config_file):
            print(f'Model {model} is not trained!')
            continue
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if not os.path.exists(mat_file) or not os.path.exists(npz_file):
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
    models_dir = defaults().PATH.MODELS
    for group in groups:
        for i in range(len(group)):
            # load mat scores
            model = group[i]['model']
            filepath = os.path.join(models_dir, model, 'scores.mat')
            scores = scipy.io.loadmat(filepath)
            pesq = scores['pesqs']
            pesq_oracle = scores['pesqs_oracle']
            stoi = scores['stois']
            stoi_oracle = scores['stois_oracle']

            # load npz scores
            filepath = os.path.join(models_dir, model, 'scores.npz')
            scores = np.load(filepath)
            mse = scores['mse']
            mse_oracle = np.zeros(mse.shape)
            seg = scores['seg']
            seg_oracle = scores['seg_oracle']

            # assign
            group[i]['pesq'] = pesq
            group[i]['stoi'] = stoi
            group[i]['mse'] = mse
            group[i]['segSSNR'] = seg[:, :, :, 0]
            group[i]['segBR'] = seg[:, :, :, 1]
            group[i]['segNR'] = seg[:, :, :, 2]
            group[i]['segRR'] = seg[:, :, :, 3]
            group[i]['oracle'] = {}
            group[i]['oracle']['pesq'] = pesq_oracle
            group[i]['oracle']['stoi'] = stoi_oracle
            group[i]['oracle']['mse'] = mse_oracle
            group[i]['oracle']['segSSNR'] = seg_oracle[:, :, :, 0]
            group[i]['oracle']['segBR'] = seg_oracle[:, :, :, 1]
            group[i]['oracle']['segNR'] = seg_oracle[:, :, :, 2]
            group[i]['oracle']['segRR'] = seg_oracle[:, :, :, 3]

            # load loss curves
            filepath = os.path.join(models_dir, model, 'losses.npz')
            curves = np.load(filepath)
            group[i]['train_curve'] = curves['train']
            group[i]['val_curve'] = curves['val']


def sort_groups_by(groups, metric):
    groups_mean_pesq = []
    for group in groups:
        mean_pesqs = [model[metric].mean() for model in group]
        group_mean_pesq = np.mean(mean_pesqs)
        groups_mean_pesq.append(group_mean_pesq)
    indexes = np.argsort(groups_mean_pesq)
    groups = [groups[i] for i in indexes]
    if metric == 'mse':
        groups = groups[::-1]
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
    models_dir = defaults().PATH.MODELS
    snrss = []
    roomss = []
    for model in models:
        # load npz scores
        filepath = os.path.join(models_dir, model, 'scores.npz')
        scores = np.load(filepath)
        snrss.append(scores['snrs'].tolist())
        roomss.append(scores['rooms'].tolist())
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


def fit_plots(n, aspect=(16, 9)):
    width = aspect[0]
    height = aspect[1]
    area = width*height*1.0
    factor = (n/area)**(1/2.0)
    cols = math.floor(width*factor)
    rows = math.floor(height*factor)
    row_first = width < height
    while rows*cols < n:
        if row_first:
            rows += 1
        else:
            cols += 1
        row_first = not(row_first)
    return rows, cols


def remove_patches(fig, axes):
    if args.format != 'svg':
        return
    for ax in axes:
        ax.patch.set_visible(False)
    fig.patch.set_visible(False)


def main(models, args, filter_):
    if args.default:
        set_default_parameters(filter_, args.dims, args.group_by)

    models = paths_to_dirnames(models)
    possible_models = find_model(**filter_)
    models = [model for model in models if model in possible_models]

    dimensions = merge_lists(args.dims, args.group_by)

    models, values = check_models(models, dimensions)
    groups, group_values = group_by_dimension(models, values, args.group_by)
    load_scores(groups)
    if args.sort_by is not None and args.sort_by != 'dims':
        groups = sort_groups_by(groups, args.sort_by)
    elif args.sort_by == 'dims':
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

    if args.top is not None:
        groups = groups[-args.top:]

    for group in groups:
        for model in group:
            print(model['model'])

    snrs, rooms = get_snrs_and_rooms(models)

    n = sum(len(group) for group in groups)
    if args.oracle:
        n *= 2
    width = 1/(n+1)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hatch_cycle = ['', '////', '\\\\\\', 'xxxx']

    figs = {}

    # summary plot
    metrics = ['pesq', 'stoi', 'segSSNR', 'segBR']
    if args.format == 'svg':
        ylabels = [r'\$\Delta PESQ\$', r'\$\Delta STOI\$', 'segSSNR', 'segBR']
    else:
        ylabels = [r'$\Delta PESQ$', r'$\Delta STOI$', 'segSSNR', 'segBR']
    rows, cols = 1, len(metrics)
    fig, axes = plt.subplots(rows, cols, figsize=args.figsize)
    if not hasattr(axes, 'flatten'):
        axes = np.array([axes])
    remove_patches(fig, axes)
    for ax, metric, ylabel in zip(axes.flatten(), metrics, ylabels):
        model_count = 0
        for i, group in enumerate(groups):
            color = color_cycle[i % len(color_cycle)]
            hatch_count = 0
            for j, model in enumerate(group):
                datas = [model[metric]]
                if args.oracle:
                    datas.append(model['oracle'][metric])
                for k, data in enumerate(datas):
                    hatch = hatch_cycle[hatch_count % len(hatch_cycle)]
                    if metric == 'mse':
                        mean = data.mean()
                        err = None
                    else:
                        mean = data.mean()
                        err = data.std()/(data.size)**0.5
                    if ax == axes.flatten()[0]:
                        if args.legend is None:
                            label = f'{model["val"]}'
                            if k == 1:
                                label += ' - oracle'
                        else:
                            label = args.legend[model_count]
                    else:
                        label = None
                    x = (model_count - (n-1)/2)*width
                    ax.bar(x=x, height=mean, width=width, label=label,
                           color=color, hatch=hatch, yerr=err)
                    model_count += 1
                    hatch_count += 1
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        if metric == 'pesq':
            if args.ymax is not None:
                ymin, _ = ax.get_ylim()
                ax.set_ylim(ymin, args.ymax)
            if args.ymin is not None:
                _, ymax = ax.get_ylim()
                ax.set_ylim(args.ymin, ymax)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
    LegendFormatter(fig, ncol=args.ncol)
    figs['summary'] = fig

    if args.train_curve:
        fig, ax = plt.subplots(1, 1)
        remove_patches(fig, [ax])
        for i, group in enumerate(groups):
            color = color_cycle[i % len(color_cycle)]
            for j, model in enumerate(group):
                label = f'{model["val"]}'
                ax.plot(model['train_curve'], label=label, color=color)
                ax.plot(model['val_curve'], '--', color=color)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
        LegendFormatter(fig, ncol=args.ncol)
        figs['train_curve'] = fig

    if args.summary:
        plt.show()
        if args.output_dir is not None:
            for key, fig in figs.items():
                fig.savefig(f'{args.output_dir}/{key}.{args.format}')
        return

    ylabels = ['MSE', r'$\Delta PESQ$', r'$\Delta PESQ$', 'segSSNR', 'segBR',
               'segNR', 'segRR']
    metrics = ['mse', 'pesq', 'stoi', 'segSSNR', 'segBR', 'segNR', 'segRR']
    for ylabel, metric in zip(ylabels, metrics):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=args.figsize)
        remove_patches(fig, axes)
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
                    if args.oracle:
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
                            if args.legend is None:
                                label = f'{model["val"]}'
                                if k == 1:
                                    label += ' - oracle'
                            else:
                                label = args.legend[model_count]
                        else:
                            label = None
                        x = np.arange(len(mean)) + (model_count-(n-1)/2)*width
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
            if metric == 'pesq':
                if args.ymax is not None:
                    ymin, _ = ax.get_ylim()
                    ax.set_ylim(ymin, args.ymax)
                if args.ymin is not None:
                    _, ymax = ax.get_ylim()
                    ax.set_ylim(args.ymin, ymax)
            ax.grid(linestyle='dotted')
            ax.set_axisbelow(True)
        LegendFormatter(fig, ncol=args.ncol)
        figs[metric] = fig

    symbols = ['o', 's', '^', 'v', '<', '>']
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    remove_patches(fig, axes)
    fig_legend_handles = []
    fig_legend_labels = []
    for axis, (ax, labels) in enumerate(zip(
                axes[::-1],
                [rooms, snrs],
            )):
        ax_legend_handles = []
        ax_legend_labels = []
        model_count = 0
        for i, group in enumerate(groups):
            color = color_cycle[i % len(color_cycle)]
            for j, model in enumerate(group):
                x = model['segBR'].mean(axis=(axis, -1))
                y = model['segSSNR'].mean(axis=(axis, -1))
                line, = ax.plot(x, y, linestyle='--',
                                color=color)
                if axis == 0:
                    if args.legend is None:
                        label = f'{model["val"]}'
                    else:
                        label = args.legend[model_count]
                    fig_legend_handles.append(line)
                    fig_legend_labels.append(label)
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
                model_count += 1
        ax.legend(ax_legend_handles, ax_legend_labels)
        ax.set_xlabel('segBR (dB)')
        ax.set_ylabel('segSSNR (dB)')
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
    lh = fig.legend(fig_legend_handles, fig_legend_labels)
    LegendFormatter(fig, lh=lh)
    figs['segmental'] = fig

    plt.show()

    if args.output_dir is not None:
        for key, fig in figs.items():
            fig.savefig(f'{args.output_dir}/{key}.{args.format}',
                        transparent=True)


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
    parser.add_argument('--summary', action='store_true',
                        help='plot only the summary of scores')
    parser.add_argument('--figsize', nargs=2, type=float,
                        help='figure size')
    parser.add_argument('--ymax', type=float,
                        help='pesq y axis upper limits')
    parser.add_argument('--ymin', type=float,
                        help='pesq y axis lower limits')
    parser.add_argument('--train-curve', action='store_true',
                        help='plot training curves')
    parser.add_argument('--oracle', action='store_true',
                        help='plot oracle scores')
    parser.add_argument('--output_dir',
                        help='output directory where to save the figures')
    parser.add_argument('--format', default='png',
                        help='output figure format')
    filter_args, args = parser.parse_args()

    model_dirs = []
    for input_ in args.input:
        if not glob(input_):
            print(f'Model not found: {input_}')
        model_dirs += glob(input_)
    main(model_dirs, args, vars(filter_args))
