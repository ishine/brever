import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 6
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
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['font.family'] = "Liberation Serif"
plt.rcParams['mathtext.fontset'] = "stix"

FIGSIZE_HEAT = (6.30*1.15, 3.19*1.15)

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
db_labels = {
    'timit_.*': 'TIMIT',
    'libri_.*': 'LibriSpeech',
    'wsj0_.*': 'WSJ',
    'clarity_.*': 'Clarity',
    'vctk_.*': 'VCTK',
    'dcase_.*': 'TAU',
    'noisex_.*': 'NOISEX',
    'icra_.*': 'ICRA',
    'demand': 'DEMAND',
    'arte': 'ARTE',
    'surrey_.*': 'Surrey',
    'ash_.*': 'ASH',
    'bras_.*': 'BRAS',
    'catt_.*': 'CATT',
    'avil_.*': 'AVIL',
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
    fig = plt.figure(figsize=FIGSIZE_HEAT)
    outer_gs = gridspec.GridSpec(1, 3, figure=fig)
    for i_dim, (dim, vals) in enumerate(dims.items()):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, len(archs), subplot_spec=outer_gs[i_dim],
            hspace=0.05, wspace=0.025,
        )
        for i_arch in range(len(archs)):
            for i_ax, i_metric in enumerate(range(3, 6)):
                ax = fig.add_subplot(inner_gs[i_ax, i_arch])
                vmin = np.percentile(scores[..., i_metric], 5)
                vmax = np.percentile(scores[..., i_metric], 95)
                data = scores[i_arch, i_dim, :, :, i_metric]
                ax.imshow(data, vmin=vmin, vmax=vmax, cmap='OrRd')
                ticklabels = [db_labels[val] for val in vals]
                ax.set_xticks(np.arange(5))
                if i_ax == 2:
                    ax.set_xticklabels(ticklabels, rotation=35, ha='right',
                                       rotation_mode='anchor')
                else:
                    ax.set_xticklabels([])
                ax.set_yticks(np.arange(5))
                if i_arch == 0:
                    ax.set_yticklabels(ticklabels, rotation=35, ha='right',
                                       rotation_mode='anchor')
                else:
                    ax.set_yticklabels([])
                annotate_heatmap(ax, data, vmin, vmax)
                if i_ax == 0:
                    ax.set_title(arch_labels[i_arch])
                if i_dim == 0 and i_arch == 0:
                    ax.set_ylabel(_m(metrics[i_metric]), fontsize='large')

    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.5)
    fig.patch.set_visible(False)
    fig.savefig('results_heatmaps.svg', bbox_inches=0)


def annotate_heatmap(ax, data, vmin, vmax):
    for i in range(5):
        for j in range(5):
            val = data[i, j]
            loc_on_scale = (val-vmin)/(vmax-vmin)
            color = 'w' if loc_on_scale > 0.4 else 'gray'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color,
                    fontsize=5.5)


def main():
    scores, scores_ref = gather_scores()
    plot_heatmaps(scores)
    plt.show()


if __name__ == '__main__':
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)
    main()
