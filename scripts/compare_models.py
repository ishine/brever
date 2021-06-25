import os
from glob import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import to_rgb
from scipy.stats import sem

import brever.modelmanagement as bmm
from brever.config import defaults


def check_models(models, dims):
    values = []  # list of dicts of output models values along dims
    models_out = []  # output models
    training_label = None  # used to check if all models use same target label
    for model in models:
        # check if model is trained
        config_file = os.path.join(model, 'config_full.yaml')
        if not os.path.exists(config_file):
            print(f'Model {model} is not trained!')
            continue
        config = bmm.read_yaml(config_file)
        # check if model is evaluated
        scores_file = os.path.join(model, 'scores.json')
        if not os.path.exists(scores_file):
            print(f'Model {model} is not evaluated!')
            continue
        # check if all models are different in the subspace of dimensions
        if dims is None:  # if no dims just return all models
            val = model  # the value is just the model name
        else:
            val = {dim: bmm.get_config_field(config, dim) for dim in dims}
            if val in values:
                dupe_model = models_out[values.index(val)]
                dupe_path = os.path.join(dupe_model, 'config_full.yaml')
                dupe_config = bmm.read_yaml(dupe_path)
                if dupe_config == config:
                    print(
                        f'Models {model} and {dupe_model} have the exact same '
                        'configuration as given by their config_full.yaml '
                        f'file. Model {model} will be skipped.'
                    )
                else:
                    raise ValueError(
                        f'Models {model} and {dupe_model} are the same in the '
                        'subspace of specified dimensions. Consider refining '
                        'the set of dimensions to compare, further filtering '
                        'the list of models, or using the --default option to '
                        'set the rest of the parameters to their default '
                        'value.'
                    )
                continue
        # check if models all have the same training label
        if training_label is None:
            training_label = bmm.get_config_field(config, 'labels')
        else:
            if training_label != bmm.get_config_field(config, 'labels'):
                raise ValueError('All models do not use the same target label')
        values.append(val)
        models_out.append(model)
    return models_out, values


def group_by_dim(models, values, all_dims, group_dims):
    # first make groups
    groups = []  # models grouped
    group_vals = []  # values describing each group
    for model, val in zip(models, values):
        if group_dims is None:
            # if no dimension specified, make as many groups as models
            groups.append([{'model': model, 'val': val}])
            if val not in group_vals:
                group_vals.append(val)
        else:
            group_outer_val = {dim: val[dim] for dim in group_dims}
            if group_outer_val not in group_vals:
                # create a new group
                group_vals.append(group_outer_val)
                groups.append([{'model': model, 'val': val}])
            else:
                # add model to existing group
                index = group_vals.index(group_outer_val)
                groups[index].append({'model': model, 'val': val})
    # then sort each group so order is consistent across groups
    if all_dims is not None and group_dims is not None:
        for i in range(len(groups)):
            for dim in all_dims:
                if dim not in group_dims:
                    groups[i] = sorted(groups[i], key=lambda x: x['val'][dim])
    if not all(len(group) == len(groups[0]) for group in groups):
        raise ValueError('Not all groups have the same size!')
    return groups, group_vals


def load_scores(groups, test_dirs):
    for group in groups:
        for i in range(len(group)):
            model = group[i]['model']
            # load scores
            with open(os.path.join(model, 'scores.json')) as f:
                scores = json.load(f)
            group[i]['scores'] = {}
            for test_dir in test_dirs:
                if test_dir not in scores.keys():
                    raise ValueError(f'{test_dir} is not in the test paths '
                                     f'of model {model}')
                group[i]['scores'][test_dir] = scores[test_dir]
            # load loss curves
            curves = np.load(os.path.join(model, 'losses.npz'))
            group[i]['train_curve'] = curves['train']
            group[i]['val_curve'] = curves['val']


def sort_groups(groups, by, dims, group_vals, test_dirs):
    if by is None:
        return groups
    if by in ['MSE', 'PESQ', 'STOI', 'segSSNR', 'segBR', 'segNR', 'segRR']:
        mean_scores = []
        for group in groups:
            mean_score, _ = make_score_matrix(group, test_dirs, 'model', by)
            mean_scores.append(mean_score.mean())
        indexes = np.argsort(mean_scores)
        groups = [groups[i] for i in indexes]
        if by == 'MSE':
            groups = groups[::-1]
    elif by == 'dims':
        if dims is not None:
            group_vals_copy = group_vals.copy()
            for dim in reversed(list(group_vals[0].keys())):
                group_vals_sorted = sorted(group_vals_copy,
                                           key=lambda x: str(x[dim]))
                i_sorted = [group_vals_copy.index(val)
                            for val in group_vals_sorted]
                groups = [groups[i] for i in i_sorted]
                group_vals_copy = [group_vals_copy[i] for i in i_sorted]
        else:
            raise ValueError("Can't sort by dims when no dim is provided")
    else:
        raise ValueError(f'Unrecognized sorting argument, got {by}')
    return groups


def set_default_parameters(filter_, dimensions, group_by):
    default_config = defaults().to_dict()
    for key, value in filter_.items():
        if (value is None and (dimensions is None or key not in dimensions)
                and (group_by is None or key not in group_by)):
            new_value = [bmm.get_config_field(default_config, key)]
            filter_[key] = new_value


def merge_lists(dimensions, group_by):
    if group_by is not None:
        if dimensions is None:
            dimensions = group_by.copy()
        else:
            for dimension in group_by:
                if dimension not in dimensions:
                    dimensions.append(dimension)
    return dimensions


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


def remove_patches(fig, axes):
    try:
        iter(axes)
    except TypeError:
        axes.patch.set_visible(False)
    else:
        for ax in axes:
            ax.patch.set_visible(False)
    fig.patch.set_visible(False)


def barplot(datas, ax, xticklabels=None, errs=None, ylabel=None, labels=None,
            colors=None):
    # check datas type
    if isinstance(datas, (float, int)):
        datas = np.array(datas)
    if isinstance(datas, np.ndarray):
        if datas.ndim == 0:
            datas = np.array(datas)
        if datas.ndim == 1:
            datas = datas[np.newaxis, :]
        if datas.ndim == 2:
            datas = [datas[:, i] for i in range(datas.shape[1])]
        elif datas.ndim == 3:
            datas = [datas[i, :, :] for i in range(datas.shape[0])]
        else:
            raise ValueError('barplot input as a numpy array must be at most '
                             f'3D, got {datas.ndim}D')
    elif not isinstance(datas, list):
        raise ValueError('barplot input must be a numpy array or a list of '
                         'numpy arrays')
    if len(datas) == 0:
        raise ValueError("Can't barplot empty data")
    for i, data in enumerate(datas):
        if not isinstance(data, np.ndarray):
            raise ValueError('barplot input must be a numpy array or a list '
                             'of numpy arrays')
        if data.size == 0:
            raise ValueError(f'barplot input at position {i} is empty')
        if data.ndim == 1:
            datas[i] = data[np.newaxis, :]
        if data.ndim > 2:
            raise ValueError(f'barplot input at position {i} is {data.ndim}D, '
                             'must be at most 2D')
        if data.shape[0] != datas[0].shape[0]:
            raise ValueError('all barplot inputs must have the same size '
                             'along first dimension')
    # check errs type
    if errs is not None:
        if isinstance(errs, (float, int)):
            errs = np.array(errs)
        if isinstance(errs, np.ndarray):
            if errs.ndim == 0:
                errs = np.array(errs)
            if errs.ndim == 1:
                errs = errs[np.newaxis, :]
            if errs.ndim == 2:
                errs = [errs[:, i] for i in range(errs.shape[1])]
            elif errs.ndim == 3:
                errs = [errs[i, :, :] for i in range(errs.shape[0])]
            else:
                raise ValueError('errs as a numpy array must be at most '
                                 f'3D, got {errs.ndim}D')
        elif not isinstance(errs, list):
            raise ValueError('errs must be a numpy array or a list of '
                             'numpy arrays')
        if len(errs) != len(datas):
            raise ValueError(f'datas and errs must have the same length, got '
                             f'{len(datas)} and {len(errs)}')
        for i, (data, err) in enumerate(zip(datas, errs)):
            if err.ndim == 1:
                errs[i] = err[np.newaxis, :]
            if errs[i].shape != data.shape:
                raise ValueError('elements of datas and errs must have same '
                                 f'shape pair-wise, got shapes {data.shape} '
                                 f'and {errs[i].shape} at position {i}')
    # check labels
    if colors is not None and len(colors) != len(datas):
        raise ValueError('colors must have the same length as datas')
    # main
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_conditions = datas[0].shape[0]
    n_models = sum(data.shape[1] for data in datas)
    if labels is not None and len(labels) != n_models:
        print(f'Warning: the number of labels ({len(labels)}) does not match '
              f'the total number of models ({n_models})')
    bar_width = 1/(n_models+1)
    model_count = 0
    for i, data in enumerate(datas):
        if colors is None:
            color = color_cycle[i % len(color_cycle)]
        else:
            color = colors[i]
        for j in range(data.shape[1]):
            color = to_rgb(color)
            color = color + (1-j/data.shape[1], )
            offset = (model_count - (n_models-1)/2)*bar_width
            x = np.arange(n_conditions) + offset
            if errs is None:
                yerr = None
            else:
                yerr = errs[i][:, j]
            if labels is None:
                label = None
            else:
                label = labels[model_count]
            ax.bar(x, data[:, j], width=bar_width, color=color, yerr=yerr,
                   label=label)
            model_count += 1
    if xticklabels is None:
        xticklabels = np.arange(n_conditions)
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='dotted')
    ax.set_axisbelow(True)


def check_scores(groups, test_dirs, system, metric):
    models = [model for group in groups for model in group]
    for test_dir in test_dirs:
        if not all(
                    model['scores'][test_dir][system][metric] ==
                    models[0]['scores'][test_dir][system][metric]
                    for model in models
                ):
            raise ValueError('All models do not have the same '
                             f'reference scores on test dir '
                             f'{test_dir}!')


def make_score_matrix(models, test_dirs, system, metric):
    n_conditions = len(test_dirs)
    n_models = len(models)
    score = np.zeros((n_conditions, n_models))
    err = np.zeros((n_conditions, n_models))
    for i, test_dir in enumerate(test_dirs):
        for j, model in enumerate(models):
            score[i, j] = np.mean(
                model['scores'][test_dir][system][metric]
            )
            err[i, j] = sem(
                model['scores'][test_dir][system][metric]
            )
    return score, err


def set_ax_lims(ax, xmin=None, xmax=None, ymin=None, ymax=None):
    if xmin is not None:
        _, ax_xmax = ax.get_xlim()
        ax.set_xlim(xmin, ax_xmax)
    if xmax is not None:
        ax_xmin, _ = ax.get_xlim()
        ax.set_xlim(ax_xmin, xmax)
    if ymin is not None:
        _, ax_ymax = ax.get_ylim()
        ax.set_ylim(ymin, ax_ymax)
    if ymax is not None:
        ax_ymin, _ = ax.get_ylim()
        ax.set_ylim(ax_ymin, ymax)


def main(models, args, filter_):
    # add default params to filter is user requested
    if args.default:
        set_default_parameters(filter_, args.dims, args.group_by)

    # filter the models
    possible_models = bmm.find_model(**filter_)
    models = [model for model in models if model in possible_models]

    # add the group dimensions to the list of dimensions
    dims = merge_lists(args.dims, args.group_by)

    # check models validity and group by dimensions
    models, values = check_models(models, dims)
    groups, group_vals = group_by_dim(models, values, dims, args.group_by)

    # load scores
    load_scores(groups, args.test_dirs)

    # check that models all have same reference and oracle scores
    for metric in ['PESQ', 'STOI']:
        check_scores(groups, args.test_dirs, 'ref', metric)
    for metric in ['PESQ', 'STOI', 'segSSNR', 'segBR', 'segNR', 'segRR']:
        check_scores(groups, args.test_dirs, 'oracle', metric)

    # sort by either dimention or score
    groups = sort_groups(groups, args.sort_by, dims, group_vals,
                         args.test_dirs)

    # take only top groups if user requested
    if args.top is not None:
        groups = groups[-args.top:]

    # print models to be plotted
    for group in groups:
        for model in group:
            print(model['model'])

    # init stuff
    figs = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ref_metrics = ['PESQ', 'STOI']
    oracle_metrics = ['PESQ', 'STOI', 'segSSNR', 'segBR', 'segNR', 'segRR']

    # summary plot
    summary_metrics = [
        'PESQ',
        'STOI',
        'segSSNR',
        'segBR',
    ]
    fig, axes = plt.subplots(1, len(summary_metrics), figsize=args.figsize)
    remove_patches(fig, axes)
    for ax, metric in zip(axes, summary_metrics):
        scores = []
        errs = []
        labels = []
        colors = []
        # first add reference score
        if not args.no_ref:
            if metric in ref_metrics:
                score, err = make_score_matrix(
                    [groups[0][0]], args.test_dirs, 'ref', metric
                )
                score, err = score.mean(axis=0), err.mean(axis=0)
                scores.append(score)
                errs.append(err)
            else:
                scores.append(np.array([0]))
                errs.append(np.array([0]))
            labels.append('ref')
            colors.append(color_cycle[0])
        # then add oracle score
        if not args.no_oracle:
            if metric in oracle_metrics:
                score, err = make_score_matrix(
                    [groups[0][0]], args.test_dirs, 'oracle', metric
                )
                score, err = score.mean(axis=0), err.mean(axis=0)
                scores.append(score)
                errs.append(err)
            else:
                scores.append(np.array([0]))
                errs.append(np.array([0]))
            labels.append('oracle')
            colors.append(color_cycle[1])
        # finally add model scores
        for i, group in enumerate(groups):
            score, err = make_score_matrix(
                group, args.test_dirs, 'model', metric,
            )
            score, err = score.mean(axis=0), err.mean(axis=0)
            scores.append(score)
            errs.append(err)
            labels.append(str(model['val']))
            colors.append(color_cycle[(i+2) % len(color_cycle)])
        if ax != axes[0]:
            labels = None
        barplot(scores, ax, errs=errs, ylabel=metric, labels=labels,
                colors=colors)
        if metric == 'PESQ':
            set_ax_lims(ax, ymin=args.ymin_pesq, ymax=args.ymax_pesq)
    LegendFormatter(fig, ncol=args.ncol)
    figs['summary'] = fig

    # training curve plot
    if args.train_curve:
        fig, ax = plt.subplots(1, 1)
        remove_patches(fig, [ax])
        for i, group in enumerate(groups):
            color = color_cycle[(i+2) % len(color_cycle)]
            for j, model in enumerate(group):
                color = to_rgb(color)
                color = color + (1-j/len(group), )
                label = str(model['val'])
                ax.plot(model['train_curve'], label=label, color=color)
                ax.plot(model['val_curve'], '--', color=color)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
        LegendFormatter(fig, ncol=args.ncol)
        figs['train_curve'] = fig

    # if summary stop here
    if not args.summary:

        # individual metric plots
        metrics = [
            'MSE',
            'PESQ',
            'STOI',
            'segSSNR',
            'segBR',
            'segNR',
            'segRR',
        ]
        for metric in metrics:
            fig, ax = plt.subplots(1, 1, sharey=True, figsize=args.figsize)
            remove_patches(fig, ax)
            scores = []
            errs = []
            labels = []
            colors = []
            # first add reference score
            if not args.no_ref and metric in ref_metrics:
                score, err = make_score_matrix(
                    [groups[0][0]], args.test_dirs, 'ref', metric
                )
                scores.append(score)
                errs.append(err)
                labels.append('ref')
                colors.append(color_cycle[0])
            # then add oracle score
            if not args.no_oracle and metric in oracle_metrics:
                score, err = make_score_matrix(
                    [groups[0][0]], args.test_dirs, 'oracle', metric
                )
                scores.append(score)
                errs.append(err)
                labels.append('oracle')
                colors.append(color_cycle[1])
            # finally add model scores
            for i, group in enumerate(groups):
                score, err = make_score_matrix(
                    group, args.test_dirs, 'model', metric,
                )
                scores.append(score)
                errs.append(err)
                labels.append(str(model['val']))
                colors.append(color_cycle[(i+2) % len(color_cycle)])
            barplot(scores, ax, errs=errs, ylabel=metric, labels=labels,
                    colors=colors)
            LegendFormatter(fig, ncol=args.ncol)
            figs[metric] = fig
            if metric == 'PESQ':
                set_ax_lims(ax, ymin=args.ymin_pesq, ymax=args.ymax_pesq)

        # segSSNR vs segBR plot
        symbols = ['o', 's', '^', 'v', '<', '>']
        fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
        remove_patches(fig, ax)
        fig_legend_handles = []
        fig_legend_labels = []
        ax_legend_handles = []
        ax_legend_labels = []
        model_count = 0
        for i, group in enumerate(groups):
            color = color_cycle[(i+2) % len(color_cycle)]
            for j, model in enumerate(group):
                x, _ = make_score_matrix(
                    [model], args.test_dirs, 'model', 'segBR'
                )
                y, _ = make_score_matrix(
                    [model], args.test_dirs, 'model', 'segSSNR'
                )
                color = to_rgb(color)
                color = color + (1-j/len(group), )
                patch = mpatches.Patch(color=color)
                if args.legend is None:
                    label = str(model['val'])
                else:
                    label = args.legend[model_count]
                fig_legend_handles.append(patch)
                fig_legend_labels.append(label)
                for k in range(len(args.test_dirs)):
                    ax.plot(x[k], y[k], marker=symbols[k], markersize=10,
                            linestyle='', color=color)
                    if i == j == 0:
                        line = mlines.Line2D([], [], linestyle='', color='k',
                                             markersize=10, marker=symbols[k])
                        label = args.test_dirs[k]
                        ax_legend_handles.append(line)
                        ax_legend_labels.append(label)
                model_count += 1
        ax.legend(ax_legend_handles, ax_legend_labels)
        ax.set_xlabel('segBR (dB)')
        ax.set_ylabel('segSSNR (dB)')
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
        lh = fig.legend(fig_legend_handles, fig_legend_labels, loc=9)
        LegendFormatter(fig, lh=lh, ncol=args.ncol)
        figs['segmental'] = fig

    plt.show()

    if args.output_dir is not None:
        for key, fig in figs.items():
            fig.savefig(f'{args.output_dir}/{key}.{args.format}')


if __name__ == '__main__':
    parser = bmm.ModelFilterArgParser(description='compare models')
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        type=lambda x: x.rstrip('/').rstrip('\\'),
                        help='list of models to compare')
    parser.add_argument('-t', '--test_dirs', nargs='+', required=True,
                        type=lambda x: x.rstrip('/').rstrip('\\'),
                        help='list of test dirs')
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
    parser.add_argument('--ymax-pesq', type=float,
                        help='pesq y axis upper limits')
    parser.add_argument('--ymin-pesq', type=float,
                        help='pesq y axis lower limits')
    parser.add_argument('--train-curve', action='store_true',
                        help='plot training curves')
    parser.add_argument('--no-ref', action='store_true',
                        help='disable ref score plotting')
    parser.add_argument('--no-oracle', action='store_true',
                        help='disable oracle score plotting')
    parser.add_argument('--output-dir',
                        help='output directory where to save the figures')
    parser.add_argument('--format', default='png',
                        help='output figure format')
    filter_args, args = parser.parse_args()

    model_dirs = []
    for input_ in args.input:
        if not glob(input_):
            print(f'Model not found: {input_}')
        model_dirs += glob(input_)

    args.test_dirs = bmm.globbed(args.test_dirs)

    main(model_dirs, args, vars(filter_args))
