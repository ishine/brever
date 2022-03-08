import argparse
import os
import json

import numpy as np

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    dict_ = {
        'speakers': [
            'timit_.*',
            'libri_.*',
            'ieee',
            'arctic',
            'hint',
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
            'dnn': 'DNN',
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

    print('\\begin{table*}')
    print('\\centering')

    archs = ['dnn', 'convtasnet']
    for arch in archs:
        print_first_row()
        for dim, vals in dict_.items():
            print_dim_multirow(dim)
            scores, gaps = [], []
            for val in vals:
                if val == 'hint':
                    continue
                p = get_train_dset(**{dim: {val}})
                p_ref = get_train_dset(**{dim: {v for v in vals if v != val}})
                p_test = get_test_dset(**{dim: {v for v in vals if v != val}})
                m = get_model(arch, p)
                m_ref = get_model(arch, p_ref)
                scores_i = get_scores(m, p_test)
                scores_ref_i = get_scores(m_ref, p_test)
                gaps_i = scores_i/scores_ref_i*100
                scores.append(scores_i)
                gaps.append(gaps_i)
            scores = np.mean(scores, axis=0)
            gaps = np.mean(gaps, axis=0)
            print_row(1, scores, gaps)
            scores, gaps = [], []
            for val in vals:
                if val == 'hint':
                    continue
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

    print('\\caption{Average scores and generalization gaps obtained by DNN '
          'and Conv-TasNet across all folds. Delta scores indicate the '
          'difference with the unprocessed input mixture.}')
    print('\\end{table*}')


if __name__ == '__main__':
    main()
