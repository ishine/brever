import os
import itertools
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import sem
import h5py

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
DELTA_SCORES = True

DATABASES = [
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
ARCHS = [
    {
        'key': 'dnn',
        'label': 'FFNN',
    },
    {
        'key': 'convtasnet',
        'label': 'Conv-TasNet',
    },
]
METRICS = [
    'PESQ',
    'STOI',
    'SNR',
]

N_CONFIGS = 2  # number of training diversity cases
N_ARCHS = len(ARCHS)
N_METRICS = len(METRICS)
N_DIM = len(DATABASES)  # number of dimensions
N_DB = len(DATABASES[0]['databases'])  # number of databases for each dimension
assert all(len(item['databases']) == N_DB for item in DATABASES)
N_MISMATCHES = 2**N_DIM  # number of combination of mismatching dimensions

if DELTA_SCORES:
    METRIC_LABELS = [fr'$\Delta${metric}' for metric in METRICS]
else:
    METRIC_LABELS = METRICS

PLOT_SPECS = {
    'figsize': {
        'single': (7.24, 4.27),
        'double': (7.24, 4.27),
        'triple': (2.66, 4.57),
    },
    'filename': {
        'single': 'results_single_{i_seed}.pdf',
        'double': 'results_double_{i_seed}.pdf',
        'triple': 'results_triple_{i_seed}.pdf',
    },
    'legend_cols': {
        'single': 4,
        'double': 4,
        'triple': 2,
    },
    'legend_fold_cols': {
        'single': 5,
        'double': 5,
        'triple': 3,
    },
    'rect': {
        'single': (0, 0, 1, 0.91),
        'double': (0, 0, 1, 0.91),
        'triple': (0, 0, 1, 0.85),
    },
    'bbox_to_anchor': {
        'single': [0.5, 0.91],
        'double': [0.5, 0.91],
        'triple': [0.5, 0.87],
    },
    'i_dims': {
        'single': [4, 5, 6],
        'double': [1, 2, 3],
        'triple': [0],
    },
    'gs_cols': {
        'single': 3,
        'double': 3,
        'triple': 1,
    },
}


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
    return [i for i in range(N_DB) if i not in idx_list]


def n_eq_one(i, dims):
    return [[i], [i], [i]]


def n_eq_four(i, dims):
    return [complement([i])]*N_DIM


def build_test_index(index, dims):
    test_index = [complement(index[dim]) for dim in range(N_DIM)]
    for dim in dims:
        test_index[dim] = index[dim]
    return test_index


def build_test_index_alt(index, dims):
    # alternative definition of generalization gap
    test_index = [[i for i in range(N_DB)] for dim in range(N_DIM)]
    for dim in dims:
        test_index[dim] = index[dim]
    return test_index


def build_kwargs(index):
    kwargs = {}
    for dim_dbs, dbs_idx in zip(DATABASES, index):
        kwargs[dim_dbs['kwarg']] = {dim_dbs['databases'][i] for i in dbs_idx}
    return kwargs


def gather_all_scores():
    shape = (N_CONFIGS, N_MISMATCHES, N_DB, N_ARCHS, N_METRICS)
    scores = np.empty(shape)
    ref_scores = np.empty(shape)
    scores_std = np.empty(shape)
    ref_scores_std = np.empty(shape)

    for i_n, index_func in enumerate([n_eq_one, n_eq_four]):

        i_mism = 0

        for ndim in range(N_DIM):  # number of matching dimensions
            for dims in itertools.combinations(range(N_DIM), ndim):
                for i_fold in range(N_DB):

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

                    for i_arch, arch in enumerate(ARCHS):
                        m = get_model(arch['key'], train_path)
                        m_ref = get_model(arch['key'], ref_train_path)

                        mean, std = get_scores(m, test_paths)
                        scores[i_n, i_mism, i_fold, i_arch, :] = mean
                        scores_std[i_n, i_mism, i_fold, i_arch, :] = std

                        mean, std = get_scores(m_ref, test_paths)
                        ref_scores[i_n, i_mism, i_fold, i_arch, :] = mean
                        ref_scores_std[i_n, i_mism, i_fold, i_arch, :] = std

                i_mism += 1

    # last mismatch scenario: matched case
    for i_n, index_func in enumerate([n_eq_one, n_eq_four]):
        for dims in [tuple(range(N_DIM))]:
            for i_fold in range(N_DB):

                index = index_func(i_fold, dims)
                kwargs = build_kwargs(index)
                train_path = get_train_dset(**kwargs)
                test_paths = get_test_dsets(index)

                for i_arch, arch in enumerate(ARCHS):
                    m = get_model(arch['key'], train_path)

                    mean, std = get_scores(m, test_paths)
                    scores[i_n, -1, i_fold, i_arch, :] = mean
                    scores_std[i_n, -1, i_fold, i_arch, :] = std
                    ref_scores[i_n, -1, i_fold, i_arch, :] = mean
                    ref_scores_std[i_n, -1, i_fold, i_arch, :] = std

    return scores, ref_scores, scores_std, ref_scores_std


def get_scores(model, test_paths):
    filename = os.path.join(model, 'scores.hdf5')
    h5f = h5py.File(filename)
    metric_idx = [list(h5f['metrics'].asstr()).index(m) for m in METRICS]
    scores = []
    for test_path in test_paths:
        if test_path not in h5f.keys():
            msg = f'{model} not tested on {test_path}'
            if args.force:
                logging.warning(msg)
                continue
            else:
                raise ValueError(msg)
        scores.append(h5f[test_path][:, metric_idx, :])
    scores = np.concatenate(scores, axis=0)
    if DELTA_SCORES:
        scores = scores[:, :, 1] - scores[:, :, 0]
    else:
        scores = scores[:, :, 1]
    mean, std = scores.mean(axis=0), scores.std(axis=0)
    h5f.close()
    return mean, std


def get_test_dsets(index):
    test_paths = []
    for indices in itertools.product(*index):
        kwargs = {
            DATABASES[i]['kwarg']: {DATABASES[i]['databases'][idx]}
            for i, idx in enumerate(indices)
        }
        test_path = get_test_dset(**kwargs)
        test_paths.append(test_path)
    return test_paths


def plot_bars(scores, scores_ref, which):
    plot_specs = {key: val[which] for key, val in PLOT_SPECS.items()}

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['^', 'v', 's', 'd', 'p']
    facecolors = [
        lambda i_arch: (*mcolors.to_rgb(colors[i_arch]), 0.5),
        lambda i_arch: 'none',
    ]
    bar_xspace = 0.5
    arch_xspace = 0.5

    config_names = [r'$N=1$', r'$N=4$']

    ylims = homothety(np.array([
        scores[:, plot_specs['i_dims'], ...].min(axis=(0, 1, 2, 3)),
        scores_ref[:, plot_specs['i_dims'], ...].max(axis=(0, 1, 2, 3)),
    ]).T, 0.10, 0.20)
    yticks = [
        np.arange(0, 1.0 + 1e-10, 0.2),
        np.arange(-0.10, 0.20 + 1e-10, 0.05),
        np.arange(0, 10 + 1e-10, 2),
    ]
    labels = [
        lambda i_arch: ARCHS[i_arch]['label'],
        lambda i_arch: ARCHS[i_arch]['label'] + '-ref',
    ]

    x = np.arange(N_ARCHS*2).reshape(N_ARCHS, 2).astype(float)
    x += arch_xspace*np.arange(N_ARCHS).reshape(-1, 1)
    xlim = x.min() - bar_xspace - 0.5, x.max() + bar_xspace + 0.5

    fig = plt.figure(figsize=plot_specs['figsize'])
    outer_gs = gridspec.GridSpec(1, plot_specs['gs_cols'], figure=fig)
    for i_gs, i_dim in enumerate(plot_specs['i_dims']):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            N_METRICS, N_CONFIGS, subplot_spec=outer_gs[i_gs],
            hspace=0.05, wspace=0.05
        )
        for i_metric in range(N_METRICS):
            for i_config in range(N_CONFIGS):
                ax = fig.add_subplot(inner_gs[i_metric, i_config])
                for i_arch in range(N_ARCHS):

                    data = scores[i_config, i_dim, :, i_arch, i_metric]
                    data_ref = scores_ref[i_config, i_dim, :, i_arch, i_metric]

                    for i_fold in range(N_DB):
                        for is_ref, data_ in enumerate([data, data_ref]):
                            label = ARCHS[i_arch]['label']
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

                    draw_gen_gap(ax, x, i_arch, data, data_ref,
                                 ylims[i_metric])

                ax.set_xticks([])
                ax.set_xlim(xlim)
                if i_gs == 0 and i_config == 0:
                    ax.text(-0.275, 0.5, _m(METRIC_LABELS[i_metric]),
                            rotation=90, transform=ax.transAxes,
                            verticalalignment='center', fontsize='large',
                            horizontalalignment='right')
                if i_metric == 2:
                    ax.set_xlabel(_m(config_names[i_config]), fontsize='large')
                ax.set_axisbelow(True)
                ax.grid(True, axis='y')
                ax.set_yticks(yticks[i_metric])
                ax.set_ylim(ylims[i_metric])
                if i_gs != 0 or i_config != 0:
                    ax.set_yticklabels([])

    handles = [
        Line2D([0], [0], marker='o', markeredgecolor=colors[i_arch],
               markerfacecolor=facecolors[is_ref](i_arch), linestyle='',
               label=labels[is_ref](i_arch), markeredgewidth=.75)
        for i_arch, is_ref in itertools.product(range(N_ARCHS), [1, 0])
    ]
    fig.legend(handles=handles, loc='upper center',
               ncol=plot_specs['legend_cols'], fontsize='medium')
    handles = [
        Line2D([0], [0], marker=markers[i_fold], markeredgecolor='k',
               markerfacecolor='none', linestyle='', label=f'Fold {i_fold+1}',
               markeredgewidth=.75)
        for i_fold in range(N_DB)
    ]
    fig.legend(handles=flip(handles, plot_specs['legend_fold_cols']),
               loc='center', ncol=plot_specs['legend_fold_cols'],
               fontsize='medium', bbox_to_anchor=plot_specs['bbox_to_anchor'])
    fig.tight_layout(rect=plot_specs['rect'], w_pad=1.8)
    fig.patch.set_visible(False)
    fig.savefig(plot_specs['filename'].format(i_seed=args.seed),
                bbox_inches='tight', pad_inches=0)


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
        print(rf'{{{METRIC_LABELS[i_metric]}}}}}')
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
    print_block(0)
    print(r'\hline \hline')
    print_block(1)
    print(r'\hline \hline')
    print_block(2)
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
        print(rf'{{{METRIC_LABELS[i_metric]}}}}}')
        for i_fold in range(N_DB):
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
    print_block(0)
    print(r'\hline \hline')
    print_block(1)
    print(r'\hline \hline')
    print_block(2)
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

    scores, ref_scores, scores_std, ref_scores_std = gather_all_scores()

    plot_bars(scores, ref_scores, 'single')
    plot_bars(scores, ref_scores, 'double')
    plot_bars(scores, ref_scores, 'triple')
    summary_table(scores)
    fold_table(scores, scores_std)
    plt.show()
