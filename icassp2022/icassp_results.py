import itertools
import os

import numpy as np

import brever.modelmanagement as bmm


def find_dset(
            kind,
            speakers={'timit_.*'},
            rooms={'surrey_.*'},
            snr_dist_args=[-5, 10],
            target_angle_lims=[-90.0, 90.0],
            noise_types={'dcase_.*'},
            random_rms=False,
            brirs=None,
        ):
    target_angle_min, target_angle_max = target_angle_lims
    dsets, configs = {
        'train': (train_dsets, train_configs),
        'test': (test_dsets, test_configs),
    }[kind]
    return bmm.find_dataset(
        dsets=dsets,
        configs=configs,
        speakers=speakers,
        rooms=rooms,
        snr_dist_args=snr_dist_args,
        target_angle_min=target_angle_min,
        target_angle_max=target_angle_max,
        noise_types=noise_types,
        random_rms=random_rms,
        filelims_room=brirs,
    )


def find_model(**kwargs):
    return bmm.find_model(models=all_models, configs=all_configs, **kwargs)


def get_score(model, test_path, metric='MSE'):
    score_file = os.path.join(model, 'scores.json')
    data = bmm.read_json(score_file)
    if metric == 'MSE':
        return np.mean(data[test_path]['model']['MSE'])
    if metric == 'dPESQ':
        return np.mean([
            data[test_path]['model']['PESQ'][i] -
            data[test_path]['ref']['PESQ'][i]
            for i in range(len(data[test_path]['model']['PESQ']))
        ])
    if metric == 'dSTOI':
        return np.mean([
            data[test_path]['model']['STOI'][i] -
            data[test_path]['ref']['STOI'][i]
            for i in range(len(data[test_path]['model']['STOI']))
        ])
    else:
        raise ValueError(f'unrecognized metric, got {metric}')


def get_generalization_gap(dim, train_cond, test_cond, ref_cond=None,
                           seeds=[0], brirs=None):
    train_brirs = brirs or 'even'
    test_brirs = brirs or 'odd'
    ref_cond = ref_cond or test_cond
    train_dset, = find_dset('train', **{dim: train_cond}, brirs=train_brirs)
    models = find_model(train_path=[train_dset], seed=seeds)
    assert len(models) == len(seeds)
    ref_dset, = find_dset('train', **{dim: ref_cond}, brirs=train_brirs)
    ref_models = find_model(train_path=[ref_dset], seed=seeds)
    assert len(ref_models) == len(seeds)
    test_dset, = find_dset('test', **{dim: test_cond}, brirs=test_brirs)
    output = {}
    for metric in ['MSE', 'dPESQ', 'dSTOI']:
        scores = [get_score(m, test_dset, metric) for m in models]
        ref_scores = [get_score(m, test_dset, metric) for m in ref_models]
        output[metric] = {}
        output[metric]['model'] = np.mean(scores)
        output[metric]['ref'] = np.mean(ref_scores)
        output[metric]['gap'] = np.mean(scores)/np.mean(ref_scores) - 1
    return output


def dict_mean(*args):
    if isinstance(args[0], float):
        return np.mean(args)
    output = {}
    for key in args[0].keys():
        output[key] = dict_mean(*[x[key] for x in args])
    return output


def get_mean_gap(dim, train_conds, test_conds, ref_conds=None, seeds=[0],
                 brirs=None):
    if ref_conds is None:
        ref_conds = [None]*len(train_conds)
    gaps = []
    for train, test, ref in zip(train_conds, test_conds, ref_conds):
        gaps.append(get_generalization_gap(dim, train, test, ref))
    return dict_mean(*gaps)


def get_inner_corpus_gap(dim, dbase, types, invert=False):
    train_conds = [set([f'{dbase}_{t}']) for t in types]
    test_conds = [set([f'{dbase}_(?!{t}$).*']) for t in types]
    if invert:
        train_conds, test_conds = test_conds, train_conds
    return get_mean_gap(dim, train_conds, test_conds)


def get_cross_corpus_gap(dim, dbases, diversity):
    if diversity == 'low':
        train_conds, ref_conds, test_conds = [], [], []
        for db in dbases:
            train_cond = set([db])
            ref_cond = set([db_ for db_ in dbases if db_ != db])
            for db_ in dbases:
                if db_ != db:
                    train_conds.append(train_cond)
                    ref_conds.append(ref_cond)
                    test_conds.append(set([db_]))
    elif diversity == 'high':
        train_conds = [set([d for d in dbases if d != db]) for db in dbases]
        test_conds = [set([db]) for db in dbases]
        ref_conds = None
    else:
        return ValueError(f'diversity must be low or high, got {diversity}')
    return get_mean_gap(dim, train_conds, test_conds, ref_conds)


def score_fmt(scores):

    def _format(x):
        x = f'{x:.2f}'
        return x.replace('.', '&.')

    out = []
    scalings = {'MSE': 100, 'dPESQ': 10, 'dSTOI': 100}
    for metric, scores in scores.items():
        score = scores['model']
        gap = scores['gap']
        scaling = scalings[metric]
        out.append(fr'{_format(score*scaling)} ({round(gap*100):+.0f}\%)')
    out = ' & '.join(out)
    return out + r' \\'


def print_inner_corpus_results(dim, dbase, diversity):
    dict_ = {
        'speakers': {
            'timit': {
                'header': 'TIMIT',
                'exps': {
                    'low': {
                        'training': '1 speaker',
                        'testing': '629 speakers',
                        'types': [
                            'm0',
                            'f0',
                            'm1',
                            'f1',
                            'm2',
                            'f2',
                        ],
                    },
                    'mid': {
                        'training': '10 speakers',
                        'testing': '620 speakers',
                        'types': [
                            '(f[0-4]|m[0-4])',
                            '(f[5-9]|m[5-9])',
                            '(f1[0-4]|m1[0-4])',
                            '(f1[5-9]|m1[5-9])',
                            '(f2[0-4]|m2[0-4])',
                        ],
                    },
                    'high': {
                        'training': '100 speakers',
                        'testing': '530 speakers',
                        'types': [
                            '(f[0-4]?[0-9]|m[0-4]?[0-9])',
                            '(f[4-9][0-9]|m[4-9][0-9])',
                            '(f1[0-4][0-9]|m1[0-4][0-9])',
                            '(f[0-9]?[02468]|m[0-9]?[02468])',
                            '(f[0-9]?[13579]|m[0-9]?[13579])',
                        ],
                    },
                },
            },
            'libri': {
                'header': 'LibriSpeech',
                'exps': {
                    'low': {
                        'training': '1 speaker',
                        'testing': '250 speakers',
                        'types': [
                            'm0',
                            'f0',
                            'm1',
                            'f1',
                            'm2',
                            'f2',
                        ],
                    },
                    'mid': {
                        'training': '10 speakers',
                        'testing': '241 speakers',
                        'types': [
                            '(f[0-4]|m[0-4])',
                            '(f[5-9]|m[5-9])',
                            '(f1[0-4]|m1[0-4])',
                            '(f1[5-9]|m1[5-9])',
                            '(f2[0-4]|m2[0-4])',
                        ],
                    },
                    'high': {
                        'training': '100 speakers',
                        'testing': '151 speakers',
                        'types': [
                            '(f[0-4]?[0-9]|m[0-4]?[0-9])',
                            '(f[4-9][0-9]|m[4-9][0-9])',
                            '(f[0-9]?[02468]|m[0-9]?[02468])',
                            '(f[0-9]?[13579]|m[0-9]?[13579])',
                        ],
                    },
                },
            },
        },
        'noise_types': {
            'dcase': {
                'header': 'TAU',
                'exps': {
                    'low': {
                        'training': '1 noise type',
                        'testing': '9 noise types',
                        'types': [
                            'airport',
                            'bus',
                            'metro',
                            'metro_station',
                            'park',
                        ],
                    },
                },
            },
            'noisex': {
                'header': 'NOISEX',
                'exps': {
                    'low': {
                        'training': '1 noise type',
                        'testing': '14 noise types',
                        'types': [
                            'babble',
                            'buccaneer1',
                            'destroyerengine',
                            'f16',
                            'factory1',
                        ],
                    },
                },
            },
        },
        'rooms': {
            'surrey': {
                'header': 'Surrey',
                'exps': {
                    'low': {
                        'training': '1 room',
                        'testing': '4 rooms',
                        'types': [
                            'anechoic',
                            'room_a',
                            'room_b',
                            'room_c',
                            'room_d',
                        ],
                    },
                }
            },
            'ash': {
                'header': 'ASH',
                'exps': {
                    'low': {
                        'training': '1 room',
                        'testing': '38 rooms',
                        'types': [
                            'r01',
                            'r02',
                            'r03',
                            'r04',
                            'r05a?b?',
                        ],
                    },
                    'high': {
                        'training': '10 rooms',
                        'testing': '29 rooms',
                        'types': [
                            'r0[0-9]a?b?',
                            'r1[0-9]',
                            'r2[0-9]',
                            'r3[0-9]',
                            'r(00|04|08|12|16|20|24|18|32|36)',
                        ],
                    },
                },
            },
        },
    }
    header = dict_[dim][dbase]['header']
    if dbase in ['dcase', 'noisex', 'surrey'] and diversity == 'high':
        exp = dict_[dim][dbase]['exps']['low']
        training, testing = exp['testing'], exp['training']
        invert = True
    else:
        exp = dict_[dim][dbase]['exps'][diversity]
        training, testing = exp['training'], exp['testing']
        invert = False
    scores = get_inner_corpus_gap(dim, dbase, exp['types'], invert=invert)
    print(f'& {header} & {training} & {testing} & {score_fmt(scores)}')


def print_cross_corpus_results(*args, diversity='low'):
    dict_ = {
        'speakers': [
            'timit_.*',
            'libri_.*',
            'ieee',
            'arctic',
            'hint',
        ],
        'noise_types': [
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
    score_dicts = []
    for db_tuple in zip(*[dict_[d] for d in args]):
        kwargs = {dim: set([dbase]) for dim, dbase in zip(args, db_tuple)}
        train_dset, = find_dset('train', **kwargs, brirs='even')
        models = find_model(train_path=[train_dset], seed=[0])
        assert len(models) == 1
        kwargs = {dim: set([db for db in dict_[dim] if db != dbase])
                  for dim, dbase in zip(args, db_tuple)}
        ref_dset, = find_dset('train', **kwargs, brirs='even')
        ref_models = find_model(train_path=[ref_dset], seed=[0])
        assert len(ref_models) == 1
        if diversity == 'low':
            for db_tuple_ in zip(*[dict_[d] for d in args]):
                if db_tuple_ != db_tuple:
                    kwargs = {dim: set([dbase])
                              for dim, dbase in zip(args, db_tuple_)}
                    test_dset, = find_dset('test', **kwargs, brirs='odd')
                    score = {}
                    for s in ['MSE', 'dPESQ', 'dSTOI']:
                        scores = [get_score(m, test_dset, s) for m in models]
                        refs = [get_score(m, test_dset, s) for m in ref_models]
                        score[s] = {}
                        score[s]['model'] = np.mean(scores)
                        score[s]['ref'] = np.mean(refs)
                        score[s]['gap'] = np.mean(scores)/np.mean(refs) - 1
                        score_dicts.append(score)
        elif diversity == 'high':
            models, ref_models = ref_models, models
            kwargs = {dim: set([dbase]) for dim, dbase in zip(args, db_tuple)}
            test_dset, = find_dset('test', **kwargs, brirs='odd')
            score = {}
            for s in ['MSE', 'dPESQ', 'dSTOI']:
                scores = [get_score(m, test_dset, s) for m in models]
                refs = [get_score(m, test_dset, s) for m in ref_models]
                score[s] = {}
                score[s]['model'] = np.mean(scores)
                score[s]['ref'] = np.mean(refs)
                score[s]['gap'] = np.mean(scores)/np.mean(refs) - 1
                score_dicts.append(score)
    scores = dict_mean(*score_dicts)
    if diversity == 'low':
        training, testing = '1 corpus', '4 corpora'
    elif diversity == 'high':
        training, testing = '4 corpora', '1 corpus'
    else:
        raise ValueError(f'diversity must be low or high, got {diversity})')
    header = {
        'speakers': 'Speech',
        'noise_types': 'Noise',
        'rooms': 'Room'
    }
    header = '-'.join(header[arg] for arg in args)
    if len(args) == 1:
        header = 'Cross-corpus'
    elif len(args) == 2:
        header = f'{header} mismatch'
    elif len(args) == 3:
        header = 'Triple mismatch'
    print(f'& {header} & {training} & {testing} & {score_fmt(scores)}')


def print_other_experiment_results(dim):

    def _format(score):
        score, gap = score['MSE']['model']*100, score['MSE']['gap']*100
        return f'{score:.2f}'.replace('.', '&.') + fr' ({round(gap):+.0f}\%)'

    dict_ = {
        'snr_dist_args': {
            'headers': ['-5 dB', '0 dB', '5 dB', '10 dB', '-5--10 dB'],
            'vals': [
                [-5, -5],
                [0, 0],
                [5, 5],
                [10, 10],
                [-5, 10],
            ],
            'brirs': None,
        },
        'target_angle_lims': {
            'headers': ['Fixed (0°)', 'Random (-90°--90°)'],
            'vals': [
                [0.0, 0.0],
                [-90.0, 90.0],
            ],
            'brirs': 'all',
        },
        'random_rms': {
            'headers': ['Fixed speaker level', r'\gls{rms} jitter'],
            'vals': [
                False,
                True,
            ],
            'brirs': None,
        },
    }
    headers, vals = dict_[dim]['headers'], dict_[dim]['vals']
    for i in range(len(headers)):
        items = [headers[i]]
        for j in range(len(headers)):
            score = get_generalization_gap(dim, vals[j], vals[i],
                                           seeds=[0, 1, 2, 3, 4],
                                           brirs=dict_[dim]['brirs'])
            items.append(_format(score))
        print(' & '.join(items) + r'\\')


def main():
    print(r'\multirow{8}{*}{\rotatebox[origin=c]{90}{Speech}}')
    for dbase in ['timit', 'libri']:
        for diversity in ['low', 'mid', 'high']:
            print_inner_corpus_results('speakers', dbase, diversity)
        print(r'\cline{2-10}')
    print_cross_corpus_results('speakers', diversity='low')
    print_cross_corpus_results('speakers', diversity='high')
    print(r'\hline \hline')

    print(r'\multirow{6}{*}{\rotatebox[origin=c]{90}{Noise}}')
    for dbase in ['dcase', 'noisex']:
        for diversity in ['low', 'high']:
            print_inner_corpus_results('noise_types', dbase, diversity)
        print(r'\cline{2-10}')
    print_cross_corpus_results('noise_types', diversity='low')
    print_cross_corpus_results('noise_types', diversity='high')
    print(r'\hline \hline')

    print(r'\multirow{6}{*}{\rotatebox[origin=c]{90}{Room}}')
    for dbase in ['surrey', 'ash']:
        for diversity in ['low', 'high']:
            print_inner_corpus_results('rooms', dbase, diversity)
        print(r'\cline{2-10}')
    print_cross_corpus_results('rooms', diversity='low')
    print_cross_corpus_results('rooms', diversity='high')
    print(r'\hline \hline')

    print(r'\multirow{8}{*}{\rotatebox[origin=c]{90}{Combined}}')
    dims = ['speakers', 'noise_types', 'rooms']
    for args in itertools.combinations(dims, 2):
        for diversity in ['low', 'high']:
            print_cross_corpus_results(*args, diversity=diversity)
        print(r'\cline{2-10}')
    for diversity in ['low', 'high']:
        print_cross_corpus_results(*dims, diversity=diversity)
    print('')

    print_other_experiment_results('snr_dist_args')
    print('')
    print_other_experiment_results('target_angle_lims')
    print('')
    print_other_experiment_results('random_rms')
    print('')


if __name__ == '__main__':
    all_models, all_configs = bmm.find_model(return_configs=True)
    train_dsets, train_configs = bmm.find_dataset('train', return_configs=True)
    test_dsets, test_configs = bmm.find_dataset('test', return_configs=True)
    main()
