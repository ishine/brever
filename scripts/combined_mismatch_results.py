import os
import json
import random
import itertools
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer
from brever.display import pretty_table

plt.rcParams['svg.fonttype'] = 'none'
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
plt.rcParams['font.family'] = "Liberation Serif"
plt.rcParams['mathtext.fontset'] = "stix"


RAW_MATH = False

dim_dict = {
    'speakers': [
        'timit_.*',
        'libri_.*',
        'wsj0_.*',
        'clarity_.*',
        'vctk_.*',
    ],
    'noises': [
        'dcase_.*',
        'noisex_.*',
        'icra_.*',
        'demand',
        'arte',
    ],
    'rooms': [
        'surrey_.*',
        'ash_.*',
        'bras_.*',
        'catt_.*',
        'avil_.*',
    ],
}
dim_labels = ['Speech', 'Noise', 'Room']
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
    kwargs = {}
    if arch == 'convtasnet-k=2':
        arch = 'convtasnet'
        kwargs['sources'] = ['foreground', 'background']
    return model_init.get_path_from_kwargs(
        arch=arch,
        train_path=arg_type_path(train_path),
        **kwargs,
    )


def get_scores(model, test_path):
    if isinstance(test_path, list):
        out = []
        for p in test_path:
            out.append(get_scores(model, p))
        return np.mean(out, axis=0)
    score_file = os.path.join(model, 'scores.json')
    with open(score_file) as f:
        scores = json.load(f)
    pesq = np.mean(scores[test_path]['model']['PESQ'])
    stoi = np.mean(scores[test_path]['model']['STOI'])
    snr = np.mean(scores[test_path]['model']['SNR'])
    pesq_i = pesq - np.mean(scores[test_path]['ref']['PESQ'])
    stoi_i = stoi - np.mean(scores[test_path]['ref']['STOI'])
    snr_i = snr - np.mean(scores[test_path]['ref']['SNR'])
    return np.array([pesq, stoi, snr, pesq_i, stoi_i, snr_i])


def get_crossval_dsets(dim, vals, i, j):
    p = get_train_dset(**{dim: {vals[j]}})
    p_ref = get_train_dset(**{dim: {v for v in vals if v != vals[j]}})
    p_test = get_test_dset(**{dim: {vals[i]}})
    return p, p_ref, p_test


def gather_scores_single_mismatch():
    scores_shape = (
        len(archs),
        len(dim_dict),
        len(list(dim_dict.values())[0]),
        len(list(dim_dict.values())[0]),
        len(metrics)
    )
    out = np.empty(scores_shape)
    out_ref = np.empty(scores_shape)
    for i_arch, arch in enumerate(archs):
        for i_dim, (dim, vals) in enumerate(dim_dict.items()):
            matrix = np.empty((5, 5, 6))
            matrix_ref = np.empty((5, 5, 6))
            for i in range(5):
                for j in range(5):
                    p, p_ref, p_test = get_crossval_dsets(dim, vals, i, j)
                    m = get_model(arch, p)
                    m_ref = get_model(arch, p_ref)
                    matrix[i, j, :] = get_scores(m, p_test)
                    matrix_ref[i, j, :] = get_scores(m_ref, p_test)
            out[i_arch, i_dim] = matrix
            out_ref[i_arch, i_dim] = matrix_ref
    return out, out_ref


def gather_scores_double_mismatch():
    dict_ = copy.deepcopy(dim_dict)
    random.seed(0)
    for dim in dict_.keys():
        random.shuffle(dict_[dim])
    scores_shape = (
        len(archs),
        len(dict_),
        len(list(dict_.values())[0]),
        len(list(dict_.values())[0]),
        len(metrics)
    )
    out = np.empty(scores_shape)
    out_ref = np.empty(scores_shape)
    for i_dim, dims in enumerate(itertools.combinations(dict_.keys(), 2)):
        for j, vals in enumerate(zip(dict_[dims[0]], dict_[dims[1]])):
            kwargs = {dim: {val} for dim, val in zip(dims, vals)}
            p = get_train_dset(**kwargs)
            kwargs = {
                dims[0]: {v for v in dict_[dims[0]] if v != vals[0]},
                dims[1]: {v for v in dict_[dims[1]] if v != vals[1]},
            }
            p_ref = get_train_dset(**kwargs)
            for i, vals_ in enumerate(zip(dict_[dims[0]], dict_[dims[1]])):
                kwargs = {dim: {val} for dim, val in zip(dims, vals_)}
                p_test = get_test_dset(**kwargs)
                for i_arch, arch in enumerate(archs):
                    m = get_model(arch, p)
                    m_ref = get_model(arch, p_ref)
                    out[i_arch, i_dim, i, j, :] = get_scores(m, p_test)
                    out_ref[i_arch, i_dim, i, j, :] = get_scores(m_ref, p_test)
    return out, out_ref


def gather_scores_triple_mismatch():
    dict_ = copy.deepcopy(dim_dict)
    random.seed(0)
    for dim in dict_.keys():
        random.shuffle(dict_[dim])
    random.seed(42)
    for dim in dict_.keys():
        random.shuffle(dict_[dim])
    scores_shape = (
        len(archs),
        1,  # len(dict_),
        len(list(dict_.values())[0]),
        len(list(dict_.values())[0]),
        len(metrics)
    )
    out = np.empty(scores_shape)
    out_ref = np.empty(scores_shape)
    for j, vals in enumerate(zip(*dict_.values())):
        kwargs = {dim: {val} for dim, val in zip(dict_.keys(), vals)}
        p = get_train_dset(**kwargs)
        kwargs = {
            dim: {v for v in dict_[dim] if v != val}
            for dim, val in zip(dict_.keys(), vals)
        }
        p_ref = get_train_dset(**kwargs)
        for i, vals_ in enumerate(zip(*dict_.values())):
            kwargs = {dim: {val} for dim, val in zip(dict_.keys(), vals_)}
            p_test = get_test_dset(**kwargs)
            for i_arch, arch in enumerate(archs):
                m = get_model(arch, p)
                m_ref = get_model(arch, p_ref)
                out[i_arch, 0, i, j, :] = get_scores(m, p_test)
                out_ref[i_arch, 0, i, j, :] = get_scores(m_ref, p_test)
    return out, out_ref


def plot_bars(scores, scores_ref, which):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hatch = ['', '////']
    ylims = [
        (0, 0.79),
        (0, 0.17),
        (0, 10.4),
    ]
    yticks = [
        np.arange(0, 0.8 + 1e-10, 0.1),
        np.arange(0, 0.16 + 1e-10, 0.03),
        np.arange(0, 10 + 1e-10, 2),
    ]
    bar_xspace = 1

    config_names = [r'$N=1$', r'$N=4$']
    figsize = {
        'single': (7.24, 4.07),
        'double': (7.24, 4.07),
        'triple': (2.66, 4.20),
    }[which]
    filename = {
        'single': 'results_single.svg',
        'double': 'results_double.svg',
        'triple': 'results_triple.svg',
    }[which]
    ncol = {
        'single': 4,
        'double': 4,
        'triple': 2,
    }[which]
    rect = {
        'single': (0, 0, 1, 0.95),
        'double': (0, 0, 1, 0.95),
        'triple': (0, 0, 1, 0.92),
    }[which]
    loc = {
        'single': 'upper center',
        'double': 'upper center',
        'triple': (0.2275, 0.91),
    }[which]

    x = np.arange(len(archs)*2).reshape(len(archs), 2)
    x += bar_xspace*np.arange(len(archs)).reshape(-1, 1)
    xlim = x.min() - bar_xspace - 0.5, x.max() + bar_xspace + 0.5

    fig = plt.figure(figsize=figsize)
    outer_gs = gridspec.GridSpec(1, scores.shape[1], figure=fig)
    for i_dim in range(scores.shape[1]):
        dim, vals = list(dim_dict.items())[i_dim]
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=outer_gs[i_dim], hspace=0.05, wspace=0.05
        )
        for i_ax, i_metric in enumerate(range(3, 6)):
            for i_config in range(2):
                ax = fig.add_subplot(inner_gs[i_ax, i_config])
                for i_arch in range(len(archs)):

                    data = scores[i_arch, i_dim, :, :, i_metric]
                    data_ref = scores_ref[i_arch, i_dim, :, :, i_metric]

                    if i_config == 0:
                        data = (data.sum(axis=0) - np.diag(data))/4
                        data_ref = (data_ref.sum(axis=0) - np.diag(data_ref))/4
                    else:
                        data, data_ref = np.diag(data_ref), np.diag(data)

                    for is_ref, data_ in enumerate([data, data_ref]):
                        label = arch_labels[i_arch]
                        label = label + '-ref' if is_ref else label
                        ax.bar(x[i_arch, is_ref], data_.mean(),
                               color=colors[i_arch], hatch=hatch[is_ref],
                               width=1, label=label, edgecolor='black',
                               yerr=data_.std())

                    draw_gen_gap(ax, x, i_arch, data, data_ref, ylims[i_ax])

                ax.set_xticks([])
                ax.set_xlim(xlim)
                if i_dim == 0 and i_config == 0:
                    # ax.set_ylabel(_m(metrics[i_metric]), fontsize='large')
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
                if i_dim != 0 or i_config != 0:
                    ax.set_yticklabels([])

        handles, labs = ax.get_legend_handles_labels()
    fig.legend(handles, labs, loc=loc, ncol=ncol, fontsize='medium')
    fig.tight_layout(rect=rect, w_pad=1.8)
    fig.patch.set_visible(False)
    fig.savefig(filename, bbox_inches=0)


def draw_gen_gap(ax, x, i_arch, data, data_ref, ylims):
    head_length = (ylims[1]-ylims[0])*0.027
    head_width = 0.22
    x = x[i_arch, 1] - 1.5
    y = data_ref.mean()
    dx = 0
    dy = data.mean() - data_ref.mean() + head_length
    dy = min(-1e-3, dy)
    ax.arrow(x, y, dx, dy, head_length=head_length, head_width=head_width,
             fc='k', length_includes_head=True, linewidth=.5)
    G_e = ((data-data_ref)/data_ref).mean()
    G_e = rf'{round(100*G_e)}%'
    ax.annotate(G_e, (x+0.3, y+head_length*1.5), ha='center')


def summary_table(single, single_ref, double, double_ref, triple, triple_ref):

    def mean_diag(data, i_arch, i_metric):
        mean = 0
        for i_dim in range(data.shape[1]):
            exp = data[i_arch, i_dim, :, :, i_metric]
            diag = np.diag(exp)
            mean += diag.mean()
        mean /= data.shape[1]
        return mean

    def mean_anti_diag(data, i_arch, i_metric):
        mean = 0
        for i_dim in range(data.shape[1]):
            exp = data[i_arch, i_dim, :, :, i_metric]
            diag = np.diag(exp)
            mean += (exp.sum() - diag.sum())/(exp.size - diag.size)
        mean /= data.shape[1]
        return mean

    def print_cell(x, y):
        if x > y:
            x = rf'\textbf{{{x:.2f}}}'
            y = f'{y:.2f}'
        else:
            x = f'{x:.2f}'
            y = rf'\textbf{{{y:.2f}}}'
        print(rf'& {x} & {y}', end=' ')

    def print_row_match(data, data_ref, i_metric, header, indent):
        for i in range(indent):
            print('& ', end=' ')
        print(rf'{header}', end=' ')
        x = mean_diag(data, 0, i_metric)
        y = mean_diag(data, 1, i_metric)
        print_cell(x, y)
        x = mean_anti_diag(data_ref, 0, i_metric)
        y = mean_anti_diag(data_ref, 1, i_metric)
        print_cell(x, y)
        print(r'\\')

    def print_row_mismatch(data, data_ref, i_metric, header, indent):
        for i in range(indent):
            print('& ', end=' ')
        print(rf'{header}', end=' ')
        x = mean_anti_diag(data, 0, i_metric)
        y = mean_anti_diag(data, 1, i_metric)
        print_cell(x, y)
        x = mean_diag(data_ref, 0, i_metric)
        y = mean_diag(data_ref, 1, i_metric)
        print_cell(x, y)
        print(r'\\')

    def print_block(i_metric):
        print(r'\multirow{6}{*}{\rotatebox[origin=c]{90}', end='')
        print(rf'{{{metrics[i_metric]}}}}}')
        print(r'& \multirow{3}{*}{\rotatebox[origin=c]{90}{Match}}')
        print_row_match(single, single_ref, i_metric, 'Single', 1)
        print_row_match(double, double_ref, i_metric, 'Double', 2)
        print_row_match(triple, triple_ref, i_metric, 'Triple', 2)
        print(r'\cline{2-7}')
        print(r'& \multirow{3}{*}{\rotatebox[origin=c]{90}{Mism.}}')
        print_row_mismatch(single, single_ref, i_metric, 'Single', 1)
        print_row_mismatch(double, double_ref, i_metric, 'Double', 2)
        print_row_mismatch(triple, triple_ref, i_metric, 'Triple', 2)

    print(r'\begin{table}')
    print(r'\centering')
    print(r'\begin{tabular}{ccccccc}')
    print(r'\hline \hline')
    print(r'& & & \multicolumn{2}{c}{$N=1$} & \multicolumn{2}{c}{$N=4$} \\')
    print(r'& & & FFNN & Conv-TasNet & FFNN & Conv-TasNet \\')

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

    def calc_mean(data, i_arch, i_metric):
        mean = 0
        for i_dim in range(data.shape[1]):
            exp = data[i_arch, i_dim, :, :, i_metric+3]
            diag = np.diag(exp)
            mean += diag.mean()
        mean /= data.shape[1]
        return f'{mean:.2f}'


def main():
    scores_1, scores_ref_1 = gather_scores_single_mismatch()
    # plot_bars(scores_1, scores_ref_1, 'single')
    scores_2, scores_ref_2 = gather_scores_double_mismatch()
    # plot_bars(scores_2, scores_ref_2, 'double')
    scores_3, scores_ref_3 = gather_scores_triple_mismatch()
    # plot_bars(scores_3, scores_ref_3, 'triple')
    summary_table(
        scores_1, scores_ref_1,
        scores_2, scores_ref_2,
        scores_3, scores_ref_3,
    )
    plt.show()


if __name__ == '__main__':
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)
    main()
