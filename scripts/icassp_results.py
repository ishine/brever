import os

import numpy as np

import brever.modelmanagement as bmm


def find_dset(
            dsets=None,
            configs=None,
            kind=None,
            speakers={'timit_.*'},
            rooms={'surrey_.*'},
            snr_dist_args=[-5, 10],
            target_angle_lims=[0.0, 0.0],
            noise_types={'dcase_.*'},
            random_rms=False,
        ):
    target_angle_min, target_angle_max = target_angle_lims
    return bmm.find_dataset(
        dsets=dsets,
        configs=configs,
        kind=kind,
        speakers=speakers,
        rooms=rooms,
        snr_dist_args=snr_dist_args,
        target_angle_min=target_angle_min,
        target_angle_max=target_angle_max,
        noise_types=noise_types,
        random_rms=random_rms,
    )


def get_score(model, test_path, score='MSE'):
    score_file = os.path.join(model, 'scores.json')
    data = bmm.read_json(score_file)
    if score == 'MSE':
        return np.mean(data[test_path]['model']['MSE'])
    if score == 'dPESQ':
        return np.mean([
            data[test_path]['model']['PESQ'][i] -
            data[test_path]['ref']['PESQ'][i]
            for i in range(len(data[test_path]['model']['PESQ']))
        ])
    if score == 'dSTOI':
        return np.mean([
            data[test_path]['model']['STOI'][i] -
            data[test_path]['ref']['STOI'][i]
            for i in range(len(data[test_path]['model']['STOI']))
        ])
    else:
        raise ValueError(f'unrecognized score, got {score}')


models, configs = bmm.find_model(return_configs=True)
train_dsets, train_configs = bmm.find_dataset('train', return_configs=True)
test_dsets, test_configs = bmm.find_dataset('test', return_configs=True)


def get_generalization_gap(
            dim,
            model_dim_val,
            cond_dim_val,
            ref_dim_val=None,
            seed=[0],
        ):
    # print(model_dim_val, cond_dim_val)
    # print(ref_dim_val, cond_dim_val)
    if ref_dim_val is None:
        ref_dim_val = cond_dim_val
    print(model_dim_val, cond_dim_val)
    print(ref_dim_val, cond_dim_val)
    train_dset, = find_dset(
        dsets=train_dsets,
        configs=train_configs,
        **{dim: model_dim_val}
    )
    model, = bmm.find_model(
        models=models,
        configs=configs,
        train_path=[train_dset],
        seed=seed,
    )
    train_dset_ref, = find_dset(
        dsets=train_dsets,
        configs=train_configs,
        **{dim: ref_dim_val}
    )
    model_ref, = bmm.find_model(
        models=models,
        configs=configs,
        train_path=[train_dset_ref],
        seed=seed,
    )
    test_dset, = find_dset(
        dsets=test_dsets,
        configs=test_configs,
        **{dim: cond_dim_val}
    )
    gap = []
    for score_name in ['MSE', 'dPESQ', 'dSTOI']:
        score = get_score(model, test_dset, score_name)
        score_ref = get_score(model_ref, test_dset, score_name)
        gap.append(score)
        gap.append((score - score_ref)/score_ref*100)
    return gap


def get_mean_gap(
            dim,
            model_dim_vals,
            cond_dim_vals,
        ):
    gaps = []
    for model_dim_val, cond_dim_val in zip(
                model_dim_vals, cond_dim_vals
            ):
        gaps.append(get_generalization_gap(
            dim,
            model_dim_val,
            cond_dim_val,
        ))
    return np.mean(gaps, axis=0)


def get_mean_gap_cross_corpus_naive(
            dim,
            corpora,
        ):
    gaps = []
    for model_dim_val in corpora:
        ref_dim_val = set(
            x.copy().pop() for x in corpora if x != model_dim_val
        )
        for cond_dim_val in ref_dim_val:
            gaps.append(get_generalization_gap(
                dim,
                model_dim_val,
                set([cond_dim_val]),
                ref_dim_val,
            ))
    return np.mean(gaps, axis=0)


def get_mean_gap_cross_corpus_fair(
            dim,
            corpora,
        ):
    model_dim_vals = [set(
            x.copy().pop() for x in corpora if x != y
        ) for y in corpora]
    return get_mean_gap(dim, model_dim_vals, corpora)


def significant_figures(x, n):
    return np.round(x, -int(np.floor(np.log10(x))) + (n - 1)) 


def timit_naive():
    gaps = get_mean_gap(
        'speakers',
        [
            {'timit_m0'},  # male 0
            {'timit_f0'},  # female 0
            {'timit_m1'},  # male 1
            {'timit_f1'},  # female 1
            {'timit_m2'},  # male 2
            {'timit_f2'},  # female 2
        ],
        [
            {'timit_(?!m0$).*'},  # male 0
            {'timit_(?!f0$).*'},  # female 0
            {'timit_(?!m1$).*'},  # male 1
            {'timit_(?!f1$).*'},  # female 1
            {'timit_(?!m2$).*'},  # male 2
            {'timit_(?!f2$).*'},  # female 2
        ],
    )
    print(
        'TIMIT & 1 speaker & 629 speakers & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def timit_fair():
    gaps = get_mean_gap(
        'speakers',
        [
            {'timit_(f[0-4]|m[0-4])'},  # males and females 0 to 4
            {'timit_(f[5-9]|m[5-9])'},  # males and females 5 to 9
            {'timit_(f1[0-4]|m1[0-4])'},  # males and females 10 to 14
            {'timit_(f1[5-9]|m1[5-9])'},  # males and females 15 to 19
            {'timit_(f2[0-4]|m2[0-4])'},  # males and females 20 to 24
        ],
        [
            {'timit_(?!(f[0-4]|m[0-4])$).*'},  # males and females 0 to 4
            {'timit_(?!(f[5-9]|m[5-9])$).*'},  # males and females 5 to 9
            {'timit_(?!(f1[0-4]|m1[0-4])$).*'},  # males and females 10 to 14
            {'timit_(?!(f1[5-9]|m1[5-9])$).*'},  # males and females 15 to 19
            {'timit_(?!(f2[0-4]|m2[0-4])$).*'},  # males and females 20 to 24
        ],
    )
    print(
        'TIMIT & 10 speaker & 620 speakers & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def timit_wise():
    gaps = get_mean_gap(
        'speakers',
        [
            {'timit_(f[0-4]?[0-9]|m[0-4]?[0-9])'},  # males and females 0 to 49
            {'timit_(f[4-9][0-9]|m[4-9][0-9])'},  # males and females 49 to 99
            {'timit_(f1[0-4][0-9]|m1[0-4][0-9])'},  # males and females 100 to 149
            {'timit_(f[0-9]?[02468]|m[0-9]?[02468])'},  # even males and females 0 to 99
            {'timit_(f[0-9]?[13579]|m[0-9]?[13579])'},  # odd males and females 0 to 99
        ],
        [
            {'timit_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},  # males and females 0 to 49
            {'timit_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},  # males and females 49 to 99
            {'timit_(?!(f1[0-4][0-9]|m1[0-4][0-9])$).*'},  # males and females 100 to 149
            {'timit_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},  # even males and females 0 to 99
            {'timit_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},  # odd males and females 0 to 99
        ],
    )
    print(
        'TIMIT & 100 speaker & 530 speakers & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def libri_naive():
    gaps = get_mean_gap(
        'speakers',
        [
            {'libri_m0'},  # male 0
            {'libri_f0'},  # female 0
            {'libri_m1'},  # male 1
            {'libri_f1'},  # female 1
            {'libri_m2'},  # male 2
            {'libri_f2'},  # female 2
        ],
        [
            {'libri_(?!m0$).*'},  # male 0
            {'libri_(?!f0$).*'},  # female 0
            {'libri_(?!m1$).*'},  # male 1
            {'libri_(?!f1$).*'},  # female 1
            {'libri_(?!m2$).*'},  # male 2
            {'libri_(?!f2$).*'},  # female 2
        ],
    )
    print(
        'LibriSpeech & 1 speaker & 250 speakers & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def libri_fair():
    gaps = get_mean_gap(
        'speakers',
        [
            {'libri_(f[0-4]|m[0-4])'},  # males and females 0 to 4
            {'libri_(f[5-9]|m[5-9])'},  # males and females 5 to 9
            {'libri_(f1[0-4]|m1[0-4])'},  # males and females 10 to 14
            {'libri_(f1[5-9]|m1[5-9])'},  # males and females 15 to 19
            {'libri_(f2[0-4]|m2[0-4])'},  # males and females 20 to 24
        ],
        [
            {'libri_(?!(f[0-4]|m[0-4])$).*'},  # males and females 0 to 4
            {'libri_(?!(f[5-9]|m[5-9])$).*'},  # males and females 5 to 9
            {'libri_(?!(f1[0-4]|m1[0-4])$).*'},  # males and females 10 to 14
            {'libri_(?!(f1[5-9]|m1[5-9])$).*'},  # males and females 15 to 19
            {'libri_(?!(f2[0-4]|m2[0-4])$).*'},  # males and females 20 to 24
        ],
    )
    print(
        'LibriSpeech & 10 speaker & 241 speakers & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def libri_wise():
    gaps = get_mean_gap(
        'speakers',
        [
            {'libri_(f[0-4]?[0-9]|m[0-4]?[0-9])'},  # males and females 0 to 49
            {'libri_(f[4-9][0-9]|m[4-9][0-9])'},  # males and females 49 to 99
            {'libri_(f[0-9]?[02468]|m[0-9]?[02468])'},  # even males and females 0 to 99
            {'libri_(f[0-9]?[13579]|m[0-9]?[13579])'},  # odd males and females 0 to 99
        ],
        [
            {'libri_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},  # males and females 0 to 49
            {'libri_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},  # males and females 49 to 99
            {'libri_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},  # even males and females 0 to 99
            {'libri_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},  # odd males and females 0 to 99
        ],
    )
    print(
        'LibriSpeech & 100 speaker & 151 speakers & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def speaker_cross_corpus_naive():
    gaps = get_mean_gap_cross_corpus_naive(
        'speakers',
        [
            {'ieee'},
            {'timit_.*'},
            {'libri_.*'},
            {'arctic'},
            {'hint'},
        ]
    )
    print(
        'Cross-corpus & 1 corpus & 4 corpora & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def speaker_cross_corpus_fair():
    gaps = get_mean_gap_cross_corpus_fair(
        'speakers',
        [
            {'ieee'},
            {'timit_.*'},
            {'libri_.*'},
            {'arctic'},
            {'hint'},
        ]
    )
    print(
        'Cross-corpus & 4 corpora & 1 corpus & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def dcase_naive():
    gaps = get_mean_gap(
        'noise_types',
        [
            {'dcase_airport'},
            {'dcase_bus'},
            {'dcase_metro'},
            {'dcase_metro_station'},
            {'dcase_park'},
        ],
        [
            {'dcase_(?!airport$).*'},
            {'dcase_(?!bus$).*'},
            {'dcase_(?!metro$).*'},
            {'dcase_(?!metro_station$).*'},
            {'dcase_(?!park$).*'},
        ],
    )
    print(
        'TAU & 1 noise type & 9 noise types & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def dcase_fair():
    gaps = get_mean_gap(
        'noise_types',
        [
            {'dcase_(?!airport$).*'},
            {'dcase_(?!bus$).*'},
            {'dcase_(?!metro$).*'},
            {'dcase_(?!metro_station$).*'},
            {'dcase_(?!park$).*'},
        ],
        [
            {'dcase_airport'},
            {'dcase_bus'},
            {'dcase_metro'},
            {'dcase_metro_station'},
            {'dcase_park'},
        ],
    )
    print(
        'TAU & 9 noise types & 1 noise type & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def noise_cross_corpus_naive():
    gaps = get_mean_gap_cross_corpus_naive(
        'noise_types',
        [
            {'dcase_.*'},
            {'icra_.*'},
            {'demand'},
            {'noisex_.*'},
            {'arte'},
        ]
    )
    print(
        'Cross-corpus & 1 corpus & 4 corpora & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def noise_cross_corpus_fair():
    gaps = get_mean_gap_cross_corpus_fair(
        'noise_types',
        [
            {'dcase_.*'},
            {'icra_.*'},
            {'demand'},
            {'noisex_.*'},
            {'arte'},
        ]
    )
    print(
        'Cross-corpus & 4 corpora & 1 corpus & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def surrey_naive():
    gaps = get_mean_gap(
        'rooms',
        [
            {'surrey_anechoic'},
            {'surrey_room_a'},
            {'surrey_room_b'},
            {'surrey_room_c'},
            {'surrey_room_d'},
        ],
        [
            {'surrey_(?!anechoic$).*'},
            {'surrey_(?!room_a$).*'},
            {'surrey_(?!room_b$).*'},
            {'surrey_(?!room_c$).*'},
            {'surrey_(?!room_d$).*'},
        ],
    )
    print(
        'Surrey & 1 room & 4 rooms & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def surrey_fair():
    gaps = get_mean_gap(
        'rooms',
        [
            {'surrey_(?!anechoic$).*'},
            {'surrey_(?!room_a$).*'},
            {'surrey_(?!room_b$).*'},
            {'surrey_(?!room_c$).*'},
            {'surrey_(?!room_d$).*'},
        ],
        [
            {'surrey_anechoic'},
            {'surrey_room_a'},
            {'surrey_room_b'},
            {'surrey_room_c'},
            {'surrey_room_d'},
        ],
    )
    print(
        'Surrey & 4 rooms & 1 room & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def ash_naive():
    gaps = get_mean_gap(
        'rooms',
        [
            {'ash_r01'},
            {'ash_r02'},
            {'ash_r03'},
            {'ash_r04'},
            {'ash_r05a?b?'},
        ],
        [
            {'ash_(?!r01$).*'},
            {'ash_(?!r02$).*'},
            {'ash_(?!r03$).*'},
            {'ash_(?!r04$).*'},
            {'ash_(?!r05a?b?$).*'},
        ],
    )
    print(
        'ASH & 1 room & 38 rooms & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def ash_fair():
    gaps = get_mean_gap(
        'rooms',
        [
            {'ash_r0[0-9]a?b?'},  # 0 to 9
            {'ash_r1[0-9]'},  # 10 to 19
            {'ash_r2[0-9]'},  # 20 to 29
            {'ash_r3[0-9]'},  # 30 to 39
            {'ash_r(00|04|08|12|16|20|24|18|32|36)'},  # every 4th room from 0 to 39
        ],
        [
            {'ash_(?!r0[0-9]a?b?$).*'},  # 0 to 9
            {'ash_(?!r1[0-9]$).*'},  # 10 to 19
            {'ash_(?!r2[0-9]$).*'},  # 20 to 29
            {'ash_(?!r3[0-9]$).*'},  # 30 to 39
            {'ash_(?!r(00|04|08|12|16|20|24|18|32|36)$).*'},  # every 4th room from 0 to 39
        ],
    )
    print(
        'ASH & 10 room & 29 rooms & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def room_cross_corpus_naive():
    gaps = get_mean_gap_cross_corpus_naive(
        'rooms',
        [
            {'surrey_.*'},
            {'ash_.*'},
            {'air_.*'},
            {'catt_.*'},
            {'avil_.*'},
        ]
    )
    print(
        'Cross-corpus & 1 corpus & 4 corpora & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def room_cross_corpus_fair():
    gaps = get_mean_gap_cross_corpus_fair(
        'rooms',
        [
            {'surrey_.*'},
            {'ash_.*'},
            {'air_.*'},
            {'catt_.*'},
            {'avil_.*'},
        ]
    )
    print(
        'Cross-corpus & 4 corpora & 1 corpus & '
        fr'{gaps[0]:.2f} ({round(gaps[1]):+.0f}\%) & '
        fr'{gaps[2]:.2f} ({round(gaps[3]):+.0f}\%) & '
        fr'{gaps[4]:.2f} ({round(gaps[5]):+.0f}\%) \\ \hline'
    )


def rooms():
    print('Rooms')
    rooms = [
        {'surrey_.*'},
        {'ash_.*'},
        {'air_.*'},
        {'catt_.*'},
        {'avil_.*'},
        {'ash_.*', 'air_.*', 'catt_.*', 'avil_.*'},
        {'surrey_.*', 'air_.*', 'catt_.*', 'avil_.*'},
        {'surrey_.*', 'ash_.*', 'catt_.*', 'avil_.*'},
        {'surrey_.*', 'ash_.*', 'air_.*', 'avil_.*'},
        {'surrey_.*', 'ash_.*', 'air_.*', 'catt_.*'},
    ]
    gaps = np.zeros((3, len(rooms), len(rooms)))
    for i, cond_dim_val in enumerate(rooms):
        for j, model_dim_val in enumerate(rooms):
            if len(cond_dim_val) == 1:
                gaps[:, i, j] = get_generalization_gap(
                    'rooms',
                    model_dim_val,
                    cond_dim_val,
                )
            else:
                gaps[:, i, j] = np.mean([get_generalization_gap(
                    'rooms',
                    model_dim_val,
                    set([x]),
                    cond_dim_val,
                ) for x in cond_dim_val], axis=0)
    print(gaps)


def snr():
    print('SNR')
    snr_dist_args = [
        [-5, -5],
        [0, 0],
        [5, 5],
        [10, 10],
        [-5, 10],
    ]
    gaps = np.zeros((6, 5, 5))
    for i, cond_dim_val in enumerate(snr_dist_args):
        for j, model_dim_val in enumerate(snr_dist_args):
            gaps[:, i, j] = get_generalization_gap(
                'snr_dist_args',
                model_dim_val,
                cond_dim_val,
            )
    print(gaps)


def direction():
    print('Direction')
    target_angle_lims = [
        [0.0, 0.0],
        [-90.0, 90.0],
    ]
    gaps = np.zeros((6, 2, 2))
    for i, cond_dim_val in enumerate(target_angle_lims):
        for j, model_dim_val in enumerate(target_angle_lims):
            gaps[:, i, j] = get_generalization_gap(
                'target_angle_lims',
                model_dim_val,
                cond_dim_val,
            )
    print(gaps)


def level():
    print('Level')
    rms_jitters = [
        False,
        True,
    ]
    gaps = np.zeros((6, 2, 2))
    for i, cond_dim_val in enumerate(rms_jitters):
        for j, model_dim_val in enumerate(rms_jitters):
            gaps[:, i, j] = get_generalization_gap(
                'random_rms',
                model_dim_val,
                cond_dim_val,
            )
    print(gaps)


# timit_naive()
# timit_fair()
# timit_wise()
# libri_naive()
# libri_fair()
# libri_wise()
# speaker_cross_corpus_naive()
# speaker_cross_corpus_fair()
# dcase_naive()
# dcase_fair()
# noise_cross_corpus_naive()
# noise_cross_corpus_fair()
# surrey_naive()
# surrey_fair()
# ash_naive()
# ash_fair()
# room_cross_corpus_naive()
# room_cross_corpus_fair()

# rooms()
# snr()
direction()
# level()
