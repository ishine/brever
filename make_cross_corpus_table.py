import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 4
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

MM_TO_INCH = 0.0393701
FIGSIZE = (160*MM_TO_INCH, 90*MM_TO_INCH)

RAW_MATH = False

dims = {
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
        duration=36000,
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


def gather_scores():
    scores = np.empty((len(archs), 3, 5, 5, 6))
    scores_ref = np.empty((len(archs), 3, 5, 5, 6))
    for i_arch, arch in enumerate(archs):
        for i_dim, (dim, vals) in enumerate(dims.items()):
            matrix = np.empty((5, 5, 6))
            matrix_ref = np.empty((5, 5, 6))
            for i in range(5):
                for j in range(5):
                    p, p_ref, p_test = get_crossval_dsets(dim, vals, i, j)
                    m = get_model(arch, p)
                    m_ref = get_model(arch, p_ref)
                    matrix[i, j, :] = get_scores(m, p_test)
                    matrix_ref[i, j, :] = get_scores(m_ref, p_test)
            scores[i_arch, i_dim] = matrix
            scores_ref[i_arch, i_dim] = matrix_ref
    return scores, scores_ref


def plot_heatmaps(scores):
    title_x = [0.20, 0.525, 0.855]
    title_y = 0.95

    fig = plt.figure(figsize=FIGSIZE)
    outer_gs = gridspec.GridSpec(1, 3, figure=fig)
    for i_dim, (dim, vals) in enumerate(dims.items()):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, len(archs), subplot_spec=outer_gs[i_dim],
            hspace=0.01, wspace=0.05
        )
        for i_arch in range(len(archs)):
            for i_ax, i_metric in enumerate(range(3, 6)):
                ax = fig.add_subplot(inner_gs[i_ax, i_arch])
                vmin = np.percentile(scores[..., i_metric], 5)
                vmax = np.percentile(scores[..., i_metric], 95)
                data = scores[i_arch, i_dim, :, :, i_metric]
                ax.imshow(data, vmin=vmin, vmax=vmax, cmap='OrRd')
                ticklabels = [val.replace('_.*', '') for val in vals]
                ax.set_xticks(np.arange(5))
                ax.set_xticklabels(ticklabels, rotation=20)
                ax.set_yticks(np.arange(5))
                if i_arch == 0:
                    ax.set_yticklabels(ticklabels)
                else:
                    ax.set_yticklabels([])
                annotate_heatmap(ax, data, vmin, vmax)
                if i_ax == 0:
                    ax.set_title(arch_labels[i_arch])
                if i_dim == 0 and i_arch == 0:
                    ax.set_ylabel(_m(metrics[i_metric]), fontsize='large')

        fig.text(title_x[i_dim], title_y, dim_labels[i_dim],
                 fontsize='x-large', ha='center')

    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.6)
    fig.patch.set_visible(False)
    fig.savefig('results_heatmaps.svg', bbox_inches=0)


def annotate_heatmap(ax, data, vmin, vmax):
    for i in range(5):
        for j in range(5):
            val = data[i, j]
            loc_on_scale = (val-vmin)/(vmax-vmin)
            color = 'w' if loc_on_scale > 0.4 else 'gray'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)


def plot_bars(scores, scores_ref):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hatch = ['', '////']
    ylims = [
        (0, 0.69),
        (0, 0.15),
        (0, 9.9),
    ]
    yticks = [
        np.linspace(0, 0.6, 7),
        np.linspace(0, 0.14, 8),
        np.linspace(0, 8, 5),
    ]
    bar_xspace = 1
    title_x = [0.21, 0.525, 0.84]
    title_y = 0.95
    config_names = [r'$N=1$', r'$N=4$']

    x = np.arange(len(archs)*2).reshape(len(archs), 2)
    x += bar_xspace*np.arange(len(archs)).reshape(-1, 1)
    xlim = x.min() - bar_xspace - 0.5, x.max() + bar_xspace + 0.5

    fig = plt.figure(figsize=FIGSIZE)
    outer_gs = gridspec.GridSpec(1, 3, figure=fig)
    for i_dim, (dim, vals) in enumerate(dims.items()):
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
                    ax.set_ylabel(_m(metrics[i_metric]), fontsize='large')
                if i_ax == 2:
                    ax.set_xlabel(_m(config_names[i_config]), fontsize='large')
                ax.set_axisbelow(True)
                ax.grid(True, axis='y')
                ax.set_yticks(yticks[i_ax])
                ax.set_ylim(ylims[i_ax])
                if i_dim != 0 or i_config != 0:
                    ax.set_yticklabels([])

        fig.text(title_x[i_dim], title_y, dim_labels[i_dim],
                 fontsize='x-large', ha='center')

        handles, labs = ax.get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', ncol=len(archs)*2,
               fontsize='large')
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), w_pad=1.8)
    fig.patch.set_visible(False)
    fig.savefig('results_bars.svg', bbox_inches=0)


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


def main():
    scores, scores_ref = gather_scores()
    plot_heatmaps(scores)
    plot_bars(scores, scores_ref)
    plt.show()


if __name__ == '__main__':
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)
    main()
