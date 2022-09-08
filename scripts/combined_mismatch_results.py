import os
import json
import itertools
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import sem

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer

# plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 7
plt.rcParams['patch.linewidth'] = .5
plt.rcParams['hatch.linewidth'] = .5
plt.rcParams['lines.linewidth'] = .5
plt.rcParams['axes.linewidth'] = .4
plt.rcParams['grid.linewidth'] = .4
plt.rcParams['xtick.major.size'] = 1
plt.rcParams['xtick.major.width'] = .5
plt.rcParams['ytick.major.size'] = 1
plt.rcParams['ytick.major.width'] = .5
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.family'] = 'Liberation Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


RAW_MATH = False

databases = [
    {
        'kwarg': 'speakers',
        'databases': [
            'timit_.*',
            'libri_.*',
            'wsj0_.*',
            'clarity_.*',
            'vctk_.*',
        ],
    },
    {
        'kwarg': 'noises',
        'databases': [
            'dcase_.*',
            'noisex_.*',
            'icra_.*',
            'demand',
            'arte',
        ],
    },
    {
        'kwarg': 'rooms',
        'databases': [
            'surrey_.*',
            'ash_.*',
            'bras_.*',
            'catt_.*',
            'avil_.*',
        ],
    },
]
archs = ['dnn', 'convtasnet']
arch_labels = ['FFNN', 'Conv-TasNet']
metrics = [
    'PESQ',
    'STOI',
    'SNR',
    r'$\Delta$PESQ',
    r'$\Delta$STOI',
    r'$\Delta$SNR',
]


def _m(s):
    return s.replace('$', r'\$') if RAW_MATH else s


def flip(items, ncol):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))


def homothety(x, p1, p2):
    return x + p1*(x - x[:, [1]]) + p2*(x - x[:, [0]])


def get_train_dset(
    speakers={'timit_.*'},
    noises={'dcase_.*'},
    rooms={'surrey_.*'},
):
    return dset_init.get_path_from_kwargs(
        kind='train',
        speakers=speakers,
        noises=noises,
        rooms=rooms,
        speech_files=[0.0, 0.8],
        noise_files=[0.0, 0.8],
        room_files='even',
        duration=3*36000,
        seed=0,
    )


def get_test_dset(
    speakers={'timit_.*'},
    noises={'dcase_.*'},
    rooms={'surrey_.*'},
):
    return dset_init.get_path_from_kwargs(
        kind='test',
        speakers=speakers,
        noises=noises,
        rooms=rooms,
        speech_files=[0.8, 1.0],
        noise_files=[0.8, 1.0],
        room_files='odd',
        duration=3600,
        seed=42,
    )


def get_model(arch, train_path):
    return model_init.get_path_from_kwargs(
        arch=arch,
        train_path=arg_type_path(train_path),
        seed=args.seed,
    )


def complement(idx_list):
    return [i for i in range(5) if i not in idx_list]


def n_eq_one(i, dims):
    return [[i], [i], [i]]


def n_eq_four(i, dims):
    return [complement([i])]*3


def build_test_index(index, dims):
    test_index = [complement(index[dim]) for dim in range(3)]
    for dim in dims:
        test_index[dim] = index[dim]
    return test_index


def build_test_index_alt(index, dims):
    # alternative definition of generalization gap
    test_index = [[i for i in range(5)] for dim in range(3)]
    for dim in dims:
        test_index[dim] = index[dim]
    return test_index


def build_kwargs(index):
    kwargs = {}
    for dim_dbs, dbs_idx in zip(databases, index):
        kwargs[dim_dbs['kwarg']] = {dim_dbs['databases'][i] for i in dbs_idx}
    return kwargs


def gather_all_scores():
    # shape = (N, mismatch_scenario, folds, arch, metrics)
    shape = (2, 8, 5, 2, 6)
    scores = np.empty(shape)
    ref_scores = np.empty(shape)
    scores_std = np.empty(shape)
    ref_scores_std = np.empty(shape)

    for i_n, index_func in enumerate([n_eq_one, n_eq_four]):

        i_mism = 0

        for ndim in range(3):
            for dims in itertools.combinations(range(3), ndim):
                for i_fold in range(5):

                    train_index = index_func(i_fold, dims)
                    train_kwargs = build_kwargs(train_index)
                    train_path = get_train_dset(**train_kwargs)
                    if args.alt:
                        test_idx = build_test_index_alt(train_index, dims)
                        test_kwargs = build_kwargs(test_idx)
                        ref_train_path = get_train_dset(**test_kwargs)
                    else:
                        test_idx = build_test_index(train_index, dims)
                        test_kwargs = build_kwargs(test_idx)
                        ref_train_path = get_train_dset(**test_kwargs)

                    test_paths = get_test_dsets(test_idx)

                    for i_arch, arch in enumerate(archs):
                        m = get_model(arch, train_path)
                        m_ref = get_model(arch, ref_train_path)

                        mean, std = get_scores(m, test_paths)
                        scores[i_n, i_mism, i_fold, i_arch, :] = mean
                        scores_std[i_n, i_mism, i_fold, i_arch, :] = std

                        mean, std = get_scores(m_ref, test_paths)
                        ref_scores[i_n, i_mism, i_fold, i_arch, :] = mean
                        ref_scores_std[i_n, i_mism, i_fold, i_arch, :] = std

                i_mism += 1

    # last mismatch scenario: matched case
    for i_n, index_func in enumerate([n_eq_one, n_eq_four]):
        for dims in [(0, 1, 2)]:
            for i_fold in range(5):

                index = index_func(i_fold, dims)
                kwargs = build_kwargs(index)
                train_path = get_train_dset(**kwargs)
                test_paths = get_test_dsets(index)

                for i_arch, arch in enumerate(archs):
                    m = get_model(arch, train_path)

                    mean, std = get_scores(m, test_paths)
                    scores[i_n, -1, i_fold, i_arch, :] = mean
                    scores_std[i_n, -1, i_fold, i_arch, :] = std
                    ref_scores[i_n, -1, i_fold, i_arch, :] = mean
                    ref_scores_std[i_n, -1, i_fold, i_arch, :] = std

    return scores, ref_scores, scores_std, ref_scores_std


def get_scores(model, test_paths):
    score_file = os.path.join(model, 'scores.json')
    with open(score_file) as f:
        scores = json.load(f)
    pesq = []
    stoi = []
    snr = []
    pesq_i = []
    stoi_i = []
    snr_i = []
    for test_path in test_paths:
        if test_path not in scores.keys():
            msg = f'{model} not tested on {test_path}'
            if args.force:
                logging.warning(msg)
                continue
            else:
                raise ValueError(msg)
        pesq += scores[test_path]['model']['PESQ']
        stoi += scores[test_path]['model']['STOI']
        snr += scores[test_path]['model']['SNR']
        pesq_i += (
            np.array(scores[test_path]['model']['PESQ']) -
            np.array(scores[test_path]['ref']['PESQ'])
        ).tolist()
        stoi_i += (
            np.array(scores[test_path]['model']['STOI']) -
            np.array(scores[test_path]['ref']['STOI'])
        ).tolist()
        snr_i += (
            np.array(scores[test_path]['model']['SNR']) -
            np.array(scores[test_path]['ref']['SNR'])
        ).tolist()
    out = np.array([pesq, stoi, snr, pesq_i, stoi_i, snr_i])
    mean, std = out.mean(axis=-1), out.std(axis=-1)
    return mean, std


def get_test_dsets(index):
    test_paths = []
    for i, j, k in itertools.product(*index):
        test_path = get_test_dset(
            speakers={databases[0]['databases'][i]},
            noises={databases[1]['databases'][j]},
            rooms={databases[2]['databases'][k]},
        )
        test_paths.append(test_path)
    return test_paths


def plot_bars(scores, scores_ref, which):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # hatch = ['', '////']
    markers = ['^', 'v', 's', 'd', 'p']
    facecolors = [
        lambda i_arch: (*mcolors.to_rgb(colors[i_arch]), 0.5),
        lambda i_arch: 'none',
    ]
    bar_xspace = 0.5
    arch_xspace = 0.5

    config_names = [r'$N=1$', r'$N=4$']
    figsize = {
        'single': (7.24, 4.27),
        'double': (7.24, 4.27),
        'triple': (2.66, 4.57),
    }[which]
    filename = {
        'single': f'results_single_{args.seed}.pdf',
        'double': f'results_double_{args.seed}.pdf',
        'triple': f'results_triple_{args.seed}.pdf',
    }[which]
    legend_cols = {
        'single': 4,
        'double': 4,
        'triple': 2,
    }[which]
    legend_fold_cols = {
        'single': 5,
        'double': 5,
        'triple': 3,
    }[which]
    rect = {
        'single': (0, 0, 1, 0.91),
        'double': (0, 0, 1, 0.91),
        'triple': (0, 0, 1, 0.85),
    }[which]
    bbox_to_anchor = {
        'single': [0.5, 0.91],
        'double': [0.5, 0.91],
        'triple': [0.5, 0.87],
    }[which]
    i_dims = {
        'single': [4, 5, 6],
        'double': [1, 2, 3],
        'triple': [0],
    }[which]
    gs_cols = {
        'single': 3,
        'double': 3,
        'triple': 1,
    }[which]
    _ix = np.ix_(
        range(scores.shape[0]),
        i_dims,
        range(scores.shape[2]),
        range(scores.shape[3]),
        [3, 4, 5],
    )
    ylims = homothety(np.array([
        scores[_ix].min(axis=(0, 1, 2, 3)),
        scores_ref[_ix].max(axis=(0, 1, 2, 3)),
    ]).T, 0.10, 0.20)
    # if which == 'single':
    #     ylims[0, 0] = 0
    #     ylims[2, 0] = 0
    # elif which == 'double':
    #     ylims[1, 0] = -0.05
    # elif which == 'triple':
    #     ylims[1, 0] = -0.10
    #     ylims[2, 0] = 0
    yticks = [
        np.arange(0, 1.0 + 1e-10, 0.2),
        np.arange(-0.10, 0.20 + 1e-10, 0.05),
        np.arange(0, 10 + 1e-10, 2),
    ]
    labels = [
        lambda i_arch: arch_labels[i_arch],
        lambda i_arch: arch_labels[i_arch] + '-ref',
    ]

    x = np.arange(len(archs)*2).reshape(len(archs), 2).astype(float)
    x += arch_xspace*np.arange(len(archs)).reshape(-1, 1)
    xlim = x.min() - bar_xspace - 0.5, x.max() + bar_xspace + 0.5

    fig = plt.figure(figsize=figsize)
    outer_gs = gridspec.GridSpec(1, gs_cols, figure=fig)
    for i_gs, i_dim in enumerate(i_dims):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=outer_gs[i_gs], hspace=0.05, wspace=0.05
        )
        for i_ax, i_metric in enumerate(range(3, 6)):
            for i_config in range(2):
                ax = fig.add_subplot(inner_gs[i_ax, i_config])
                for i_arch in range(len(archs)):

                    data = scores[i_config, i_dim, :, i_arch, i_metric]
                    data_ref = scores_ref[i_config, i_dim, :, i_arch, i_metric]

                    for i_fold in range(5):
                        for is_ref, data_ in enumerate([data, data_ref]):
                            label = arch_labels[i_arch]
                            label = label + '-ref' if is_ref else label
                            ax.scatter(x[i_arch, 1-is_ref], data_[i_fold],
                                       label=labels[is_ref](i_arch), s=30,
                                       marker=markers[i_fold], linewidth=.75,
                                       fc=facecolors[is_ref](i_arch),
                                       ec=colors[i_arch])
                        ax.plot(
                            x[i_arch, ::-1],
                            [data[i_fold], data_ref[i_fold]],
                            color=colors[i_arch], ls='--', lw=.75,
                        )

                    draw_gen_gap(ax, x, i_arch, data, data_ref, ylims[i_ax])

                ax.set_xticks([])
                ax.set_xlim(xlim)
                if i_gs == 0 and i_config == 0:
                    ax.text(-0.275, 0.5, _m(metrics[i_metric]), rotation=90,
                            verticalalignment='center',
                            horizontalalignment='right',
                            transform=ax.transAxes, fontsize='large')
                if i_ax == 2:
                    ax.set_xlabel(_m(config_names[i_config]), fontsize='large')
                ax.set_axisbelow(True)
                ax.grid(True, axis='y')
                ax.set_yticks(yticks[i_ax])
                ax.set_ylim(ylims[i_ax])
                if i_gs != 0 or i_config != 0:
                    ax.set_yticklabels([])

    handles = [
        Line2D([0], [0], marker='o', markeredgecolor=colors[i_arch],
               markerfacecolor=facecolors[is_ref](i_arch), linestyle='',
               label=labels[is_ref](i_arch), markeredgewidth=.75)
        for i_arch, is_ref in itertools.product(range(len(archs)), [1, 0])
    ]
    fig.legend(handles=handles, loc='upper center', ncol=legend_cols,
               fontsize='medium')
    handles = [
        Line2D([0], [0], marker=markers[i_fold], markeredgecolor='k',
               markerfacecolor='none', linestyle='', label=f'Fold {i_fold+1}',
               markeredgewidth=.75)
        for i_fold in range(5)
    ]
    fig.legend(handles=flip(handles, legend_fold_cols), loc='center',
               ncol=legend_fold_cols,
               fontsize='medium', bbox_to_anchor=bbox_to_anchor)
    fig.tight_layout(rect=rect, w_pad=1.8)
    fig.patch.set_visible(False)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)


def draw_gen_gap(ax, x, i_arch, data, data_ref, ylims):
    # head_length = (ylims[1]-ylims[0])*0.027
    # head_width = 0.22
    x = x[i_arch].mean()
    y = max(data.max(), data_ref.max()) + (ylims[1]-ylims[0])*0.06
    # dx = 0
    # dy = data.mean() - data_ref.mean() + head_length
    # dy = min(-1e-3, dy)
    # ax.arrow(x, y, dx, dy, head_length=head_length, head_width=head_width,
    #          fc='k', length_includes_head=True, linewidth=.5)
    gg = ((data-data_ref)/data_ref).mean()
    gg_sem = sem(((data-data_ref)/data_ref))
    if np.isnan(gg):
        gg = 'NaN'
    else:
        gg = rf'{round(100*gg)}$\pm${round(100*gg_sem)}%'
        # gg = rf'{round(100*gg)}%'
    ax.annotate(gg, (x, y), ha='center')


def summary_table(scores):

    def print_cell(x, y):
        if x > y:
            x = rf'\textbf{{{x:.2f}}}'
            y = f'{y:.2f}'
        else:
            x = f'{x:.2f}'
            y = rf'\textbf{{{y:.2f}}}'
        print(rf'& {x} & {y}', end=' ')

    def print_row_match(i_metric):
        print('& ', end=' ')
        print('Match', end=' ')
        x = scores[0, -1, :, 0, i_metric].mean()
        y = scores[0, -1, :, 1, i_metric].mean()
        print_cell(x, y)
        x = scores[1, -1, :, 0, i_metric].mean()
        y = scores[1, -1, :, 1, i_metric].mean()
        print_cell(x, y)
        print(r'\\')

    def print_row_mismatch(i_metric, header, i_dims):
        print('& ', end=' ')
        print(rf'{header}', end=' ')
        x = scores[0, i_dims, :, 0, i_metric].mean()
        y = scores[0, i_dims, :, 1, i_metric].mean()
        print_cell(x, y)
        x = scores[1, i_dims, :, 0, i_metric].mean()
        y = scores[1, i_dims, :, 1, i_metric].mean()
        print_cell(x, y)
        print(r'\\')

    def print_block(i_metric):
        print(r'\multirow{4}{*}{\rotatebox[origin=c]{90}', end='')
        print(rf'{{{metrics[i_metric]}}}}}')
        print_row_match(i_metric)
        print_row_mismatch(i_metric, 'Single mism.', [4, 5, 6])
        print_row_mismatch(i_metric, 'Double mism.', [1, 2, 3])
        print_row_mismatch(i_metric, 'Triple mism.', [0])

    print(r'\begin{table}')
    print(r'\centering')
    print(r'\begin{tabular}{cccccc}')
    print(r'\hline \hline')
    print(r'& & \multicolumn{2}{c}{$N=1$} & \multicolumn{2}{c}{$N=4$} \\')
    print(r'& & FFNN & Conv-TasNet & FFNN & Conv-TasNet \\')

    print(r'\hline \hline')
    print_block(3)
    print(r'\hline \hline')
    print_block(4)
    print(r'\hline \hline')
    print_block(5)
    print(r'\hline \hline')

    print(r'\end{tabular}')
    print(r'\caption{Caption}')
    print(r'\label{tab:summary}')
    print(r'\end{table}')
    print('')


def fold_table(scores, scores_std):

    def print_cell(x, y, x_std=None, y_std=None):
        if x > y:
            x = rf'\textbf{{{x:.2f}}}'
            y = f'{y:.2f}'
        else:
            x = f'{x:.2f}'
            y = rf'\textbf{{{y:.2f}}}'
        # if x_std is not None and y_std is not None:
        if False:
            x_std = f'{x_std:.2f}'
            y_std = f'{y_std:.2f}'
            print(rf'& {x} \pm {x_std} & {y} \pm {y_std}', end=' ')
        else:
            print(rf'& {x} & {y}', end=' ')

    def print_row(i_metric, i_fold):
        print('& ', end=' ')
        print(f'Fold {i_fold+1}', end=' ')
        x = scores[0, -1, i_fold, 0, i_metric]
        y = scores[0, -1, i_fold, 1, i_metric]
        x_std = scores_std[0, -1, i_fold, 0, i_metric]
        y_std = scores_std[0, -1, i_fold, 1, i_metric]
        print_cell(x, y, x_std, y_std)
        x = scores[1, -1, i_fold, 0, i_metric]
        y = scores[1, -1, i_fold, 1, i_metric]
        x_std = scores_std[1, -1, i_fold, 0, i_metric]
        y_std = scores_std[1, -1, i_fold, 1, i_metric]
        print_cell(x, y, x_std, y_std)
        print(r'\\')

    def print_row_mean(i_metric):
        print('& ', end=' ')
        print('Mean', end=' ')
        x = scores[0, -1, :, 0, i_metric].mean()
        y = scores[0, -1, :, 1, i_metric].mean()
        print_cell(x, y)
        x = scores[1, -1, :, 0, i_metric].mean()
        y = scores[1, -1, :, 1, i_metric].mean()
        print_cell(x, y)
        print(r'\\')

    def print_block(i_metric):
        print(r'\multirow{6}{*}{\rotatebox[origin=c]{90}', end='')
        print(rf'{{{metrics[i_metric]}}}}}')
        for i_fold in range(5):
            print_row(i_metric, i_fold)
        print(r'\cline{2-6}')
        print_row_mean(i_metric)

    print(r'\begin{table}')
    print(r'\centering')
    print(r'\begin{tabular}{cccccc}')
    print(r'\hline \hline')
    print(r'& & \multicolumn{2}{c}{$N=1$} & \multicolumn{2}{c}{$N=4$} \\')
    print(r'& & FFNN & Conv-TasNet & FFNN & Conv-TasNet \\')

    print(r'\hline \hline')
    print_block(3)
    print(r'\hline \hline')
    print_block(4)
    print(r'\hline \hline')
    print_block(5)
    print(r'\hline \hline')

    print(r'\end{tabular}')
    print(r'\caption{Caption}')
    print(r'\label{tab:summary}')
    print(r'\end{table}')
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--alt', action='store_true')
    args = parser.parse_args()

    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    npzfile = 'temp.npz'
    if os.path.exists(npzfile):
        npzobj = np.load(npzfile)
        scores = npzobj['scores']
        ref_scores = npzobj['ref_scores']
        scores_std = npzobj['scores_std']
        ref_scores_std = npzobj['ref_scores_std']
    else:
        scores, ref_scores, scores_std, ref_scores_std = gather_all_scores()
        np.savez(
            npzfile,
            scores=scores,
            ref_scores=ref_scores,
            scores_std=scores_std,
            ref_scores_std=ref_scores_std,
        )

    plot_bars(scores, ref_scores, 'single')
    plot_bars(scores, ref_scores, 'double')
    plot_bars(scores, ref_scores, 'triple')
    summary_table(scores)
    fold_table(scores, scores_std)
    plt.show()
