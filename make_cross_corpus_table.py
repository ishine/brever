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
RAW_MATH = True


def _m(s):
    return s.replace("$", r"\$") if RAW_MATH else s


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    dict_ = {
        'speakers': [
            'timit_.*',
            'libri_.*',
            'ieee',
            'arctic',
            'vctk',
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
            duration=1800,
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

    def print_row(n_train, scores, gaps):
        scores = [f'{x:.2f}' for x in scores]
        gaps = [f'{x:.0f}' for x in gaps]
        cells = ['', str(n_train), str(5-n_train)]
        cells += [f'{s} ({g}\\%)' for s, g in zip(scores, gaps)]
        row = ' & '.join(cells) + ' \\\\'
        print(row)

    def print_first_row():
        print('\\begin{subtable}{\\textwidth}')
        print('\\centering')
        print('\\begin{tabular}{c|cccccccc}')
        print('\\hline\\hline')
        cells = [
            '',
            '\\makecell{Training\\\\corpora}',
            '\\makecell{Testing\\\\corpora}',
            'PESQ',
            'STOI',
            'SNR',
            '$\\Delta$PESQ',
            '$\\Delta$STOI',
            '$\\Delta$SNR',
        ]
        row = ' & '.join(cells) + ' \\\\'
        print(row)

    def print_last_row(arch):
        dict_ = {
            'dnn': 'FFNN',
            'convtasnet': 'Conv-TasNet',
        }
        arch = dict_[arch]
        print('\\hline\\hline')
        print('\\end{tabular}')
        print(f'\\caption{{{arch}}}')
        print('\\end{subtable}')
        print('\\par\\medskip')

    def print_dim_multirow(dim):
        dict_ = {
            'speakers': 'Speech',
            'noises': 'Noise',
            'rooms': 'Room',
        }
        dim = dict_[dim]
        out = f'\\hline\\multirow{{2}}{{*}}{{{dim}}}'
        print(out)

    def add_title(fig, i_dim):
        x = [0.20, 0.512, 0.819][i_dim]
        y = 1
        text = ['Speech', 'Noise', 'Room'][i_dim]
        fig.text(x, y, text)

    print('\\begin{table*}')
    print('\\centering')

    archs = ['dnn', 'convtasnet']
    for arch in archs:
        print_first_row()
        for dim, vals in dict_.items():
            print_dim_multirow(dim)
            scores, gaps = [], []
            for val in vals:
                p = get_train_dset(**{dim: {val}})
                p_ref = get_train_dset(**{dim: {v for v in vals if v != val}})
                p_tests = []
                for v in vals:
                    if v != val:
                        p_test = get_test_dset(**{dim: {val}})
                        p_tests.append(p_test)
                m = get_model(arch, p)
                m_ref = get_model(arch, p_ref)
                scores_i = get_scores(m, p_tests)
                scores_ref_i = get_scores(m_ref, p_tests)
                gaps_i = scores_i/scores_ref_i*100
                scores.append(scores_i)
                gaps.append(gaps_i)
            scores = np.mean(scores, axis=0)
            gaps = np.mean(gaps, axis=0)
            print_row(1, scores, gaps)
            scores, gaps = [], []
            for val in vals:
                p = get_train_dset(**{dim: {v for v in vals if v != val}})
                p_ref = get_train_dset(**{dim: {val}})
                p_test = get_test_dset(**{dim: {val}})
                m = get_model(arch, p)
                m_ref = get_model(arch, p_ref)
                scores_i = get_scores(m, p_test)
                scores_ref_i = get_scores(m_ref, p_test)
                gaps_i = scores_i/scores_ref_i*100
                scores.append(scores_i)
                gaps.append(gaps_i)
            scores = np.mean(scores, axis=0)
            gaps = np.mean(gaps, axis=0)
            print_row(4, scores, gaps)
        print_last_row(arch)

    print('\\caption{Average scores and generalization gaps obtained by FFNN '
          'and Conv-TasNet across all folds. Delta scores indicate the '
          'difference with the unprocessed input mixture.}')
    print('\\end{table*}')

    matrices = np.empty((2, 3, 5, 5, 6))
    matrices_ref = np.empty((2, 3, 5, 5, 6))
    for i_arch, arch in enumerate(archs):
        for i_dim, (dim, vals) in enumerate(dict_.items()):
            matrix = np.empty((5, 5, 6))
            matrix_ref = np.empty((5, 5, 6))
            for i in range(5):
                for j in range(5):
                    p = get_train_dset(**{dim: {vals[j]}})
                    p_ref = get_train_dset(**{dim: {v for v in vals if v != vals[j]}})
                    p_test = get_test_dset(**{dim: {vals[i]}})
                    m = get_model(arch, p)
                    m_ref = get_model(arch, p_ref)
                    matrix[i, j, :] = get_scores(m, p_test)
                    matrix_ref[i, j, :] = get_scores(m_ref, p_test)
            matrices[i_arch, i_dim] = matrix
            matrices_ref[i_arch, i_dim] = matrix_ref

    for i_metric in range(6):
        fig, axes = plt.subplots(3, 4)
        vmin = np.percentile(np.stack([matrices[..., i_metric], matrices_ref[..., i_metric]]), 5)
        vmax = np.percentile(np.stack([matrices[..., i_metric], matrices_ref[..., i_metric]]), 95)
        for i_dim, (dim, vals) in enumerate(dict_.items()):
            for i_arch in range(2):
                for is_ref in [0, 1]:
                    data = matrices_ref if is_ref else matrices
                    data = data[i_arch, i_dim, :, :, i_metric]
                    ax = axes[i_dim, 2*i_arch + is_ref]
                    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap='OrRd')
                    ax.set_xticks(np.arange(5))
                    ax.set_xticklabels(vals)
                    ax.set_yticks(np.arange(5))
                    ax.set_yticklabels(vals)
                    for i in range(5):
                        for j in range(5):
                            text = ax.text(j, i, f'{data[i, j]:.2f}',
                                           ha="center", va="center", color="w")
                    if i_dim == 0:
                        ax.set_title([
                            ['FFNN', 'FFNN-ref'],
                            ['Conv-TasNet', 'Conv-TasNet-ref'],
                        ][i_arch][is_ref])
                    if i_arch == 0 and not is_ref:
                        ax.set_ylabel(dim)
        fig.suptitle(['pesq', 'stoi', 'snr', 'pesq_i', 'stoi_i', 'snr_i'][i_metric])

    # for i_metric in range(6):
    #     fig, axes = plt.subplots(3, 2)
    #     vmin = 0
    #     vmax = max(matrices[..., i_metric].max(), matrices_ref[..., i_metric].max())
    #     for i_dim, (dim, vals) in enumerate(dict_.items()):
    #         for i_arch in range(2):
    #             data = matrices[i_arch, i_dim, :, :, i_metric]
    #             data_ref = matrices_ref[i_arch, i_dim, :, :, i_metric]
    #             data = (data.sum(axis=0) - np.diag(data))/4
    #             data_ref = (data_ref.sum(axis=0) - np.diag(data_ref))/4
    #             ax = axes[i_dim, i_arch]
    #             ax.bar(np.arange(5)*3, data, label='main')
    #             ax.bar(np.arange(5)*3+1, data_ref, label='ref')
    #             ax.set_xticks(np.arange(5)*3+0.5)
    #             ax.set_xticklabels(vals)
    #             ax.set_ylim(vmin, vmax)
    #             if i_dim == 0:
    #                 ax.set_title(['FFNN', 'Conv-TasNet'][i_arch])
    #             if i_arch == 0:
    #                 ax.set_ylabel(dim)
    #             ax.legend()
    #     fig.suptitle(['pesq', 'stoi', 'snr', 'pesq_i', 'stoi_i', 'snr_i'][i_metric])

    # scores = np.empty((2, 3, 2, 2, 6))  # archs, dims, lo/hi, main/ref, metrics
    # # ref score, lo div
    # tmp = []
    # for i in range(5):
    #     for j in range(5):
    #         if i != j:
    #             tmp.append(matrices_ref[:, :, i, j, :])
    # scores[:, :, 0, 1, :] = np.mean(tmp, axis=0)
    # # main score, lo div
    # tmp = []
    # for i in range(5):
    #     for j in range(5):
    #         if i != j:
    #             tmp.append(matrices[:, :, i, j, :])
    # scores[:, :, 0, 0, :] = np.mean(tmp, axis=0)
    # # ref score, hi div
    # tmp = []
    # for i in range(5):
    #     tmp.append(matrices[:, :, i, i, :])
    # scores[:, :, 1, 1, :] = np.mean(tmp, axis=0)
    # # main score, hi div
    # tmp = []
    # for i in range(5):
    #     tmp.append(matrices_ref[:, :, i, i, :])
    # scores[:, :, 1, 0, :] = np.mean(tmp, axis=0)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hatch = ['', '////', r'\\\\']
    labels = [['FFNN', 'FFNN-ref', 'FFNN-naive'], ['Conv-TasNet', 'Conv-TasNet-ref', 'Conv-TasNet-naive']]

    fig = plt.figure(figsize=(6, 3.5))
    outer_gs = gridspec.GridSpec(1, 3, figure=fig)
    for i_dim, (dim, vals) in enumerate(dict_.items()):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=outer_gs[i_dim], hspace=0.05, wspace=0.05
        )
        for i_metric in range(3):
            for i_config in range(2):
                ax = fig.add_subplot(inner_gs[i_metric, i_config])

                # sc = scores[:, i_dim, i_config, :, i_metric+3]

                x = np.array([0, 1, 2, 3.8, 4.8, 5.8])
                x = x - np.mean(x)
                x = x.reshape(2, 3)
                ymin, ymax = [
                    (0, 0.59),
                    (0, 0.14),
                    (0, 11),
                ][i_metric]
                yrange = ymax - ymin
                for arch in range(2):

                    if i_config == 0:
                        data = matrices[arch, i_dim, :, :, i_metric+3]
                        data = (data.sum(axis=0) - np.diag(data))/4
                        data_ref = matrices_ref[arch, i_dim, :, :, i_metric+3]
                        data_ref = (data_ref.sum(axis=0) - np.diag(data_ref))/4
                        data_naive = np.diag(matrices[arch, i_dim, :, :, i_metric+3])
                    else:
                        data = np.diag(matrices_ref[arch, i_dim, :, :, i_metric+3])
                        data_ref = np.diag(matrices[arch, i_dim, :, :, i_metric+3])
                        data_naive = matrices_ref[arch, i_dim, :, :, i_metric+3]
                        data_naive = (data_naive.sum(axis=0) - np.diag(data_naive))/4

                    ax.bar(x[arch, 0], data.mean(), color=color_cycle[arch],
                           hatch=hatch[0], width=1, label=labels[arch][0],
                           edgecolor='black', yerr=data.std())
                    ax.bar(x[arch, 1], data_ref.mean(), color=color_cycle[arch],
                           hatch=hatch[1], width=1, label=labels[arch][1],
                           edgecolor='black', yerr=data_ref.std())
                    ax.bar(x[arch, 2], data_naive.mean(), color=color_cycle[arch],
                           hatch=hatch[2], width=1, label=labels[arch][2],
                           edgecolor='black', yerr=data_naive.std())

                    hl = yrange*0.027
                    hw = 0.22
                    x_ = x[arch, 1] - 1.5
                    y_ = data_ref.mean()
                    dx = 0
                    dy = data.mean() - data_ref.mean() + hl
                    ax.arrow(x_, y_, dx, dy, head_length=hl, head_width=hw,
                             fc='k', length_includes_head=True, linewidth=.5)

                    G_e = ((data-data_ref)/data_ref).mean()
                    G_e = rf'{round(100*G_e)}%'
                    ax.annotate(G_e, (x_+0.3, y_+hl*1.5), ha='center')
                ax.set_xticks([])
                ax.set_xlim([-5, 5])
                if i_dim == 0 and i_config == 0:
                    ax.set_ylabel([
                        _m(r'$\Delta$PESQ'),
                        _m(r'$\Delta$STOI'),
                        _m(r'$\Delta$SNR'),
                    ][i_metric])
                if i_metric == 2:
                    ax.set_xlabel([
                        _m(r'$N=1$'),
                        _m(r'$N=4$'),
                    ][i_config])
                ax.set_axisbelow(True)
                ax.grid(True, axis='y')
                ax.set_yticks([
                    np.linspace(0, 0.5, 6),
                    np.linspace(0, 0.12, 7),
                    np.linspace(0, 10, 6),
                ][i_metric])
                ax.set_ylim(ymin, ymax)
                if i_dim != 0 or i_config != 0:
                    ax.set_yticklabels([])
        add_title(fig, i_dim)
        handles, labs = ax.get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', ncol=len(labels))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.6)
    fig.patch.set_visible(False)
    fig.savefig('../interspeech-2022-submission/results_all.svg', bbox_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
