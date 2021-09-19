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
            ref_dim_val=None
        ):
    if ref_dim_val is None:
        ref_dim_val = cond_dim_val
    train_dset, = find_dset(
        dsets=train_dsets,
        configs=train_configs,
        **{dim: model_dim_val}
    )
    model, = bmm.find_model(
        models=models,
        configs=configs,
        train_path=[train_dset],
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
        print(model_dim_val, cond_dim_val)
        gaps.append(get_generalization_gap(
            dim,
            model_dim_val,
            cond_dim_val,
        ))
    [print(np.round(gap)) for gap in gaps]
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
    [print(np.round(gap)) for gap in gaps]
    return np.mean(gaps, axis=0)


def get_mean_gap_cross_corpus_fair(
            dim,
            corpora,
        ):
    model_dim_vals = [set(
            x.copy().pop() for x in corpora if x != y
        ) for y in corpora]
    return get_mean_gap(dim, model_dim_vals, corpora)


def timit_naive():
    print('TIMIT naive')
    print(np.round(get_mean_gap(
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
    )))


def timit_fair():
    print('TIMIT fair')
    print(np.round(get_mean_gap(
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
    )))


def timit_wise():
    print('TIMIT wise')
    print(np.round(get_mean_gap(
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
    )))


def libri_naive():
    print('LibriSpeech naive')
    print(np.round(get_mean_gap(
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
    )))


def libri_fair():
    print('LibriSpeech fair')
    print(np.round(get_mean_gap(
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
    )))


def libri_wise():
    print('LibriSpeech wise')
    print(np.round(get_mean_gap(
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
    )))


def speaker_cross_corpus_naive():
    print('Speaker cross-corpus naive')
    print(np.round(get_mean_gap_cross_corpus_naive(
        'speakers',
        [
            {'ieee'},
            {'timit_.*'},
            {'libri_.*'},
            {'arctic'},
            {'hint'},
        ]
    )))


def speaker_cross_corpus_fair():
    print('Speaker cross-corpus fair')
    print(np.round(get_mean_gap_cross_corpus_fair(
        'speakers',
        [
            {'ieee'},
            {'timit_.*'},
            {'libri_.*'},
            {'arctic'},
            {'hint'},
        ]
    )))


def dcase_naive():
    print('DCASE naive')
    print(np.round(get_mean_gap(
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
    )))


def dcase_fair():
    print('DCASE fair ')
    print(np.round(get_mean_gap(
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
    )))


def noise_cross_corpus_naive():
    print('Noise cross-corpus naive')
    print(np.round(get_mean_gap_cross_corpus_naive(
        'noise_types',
        [
            {'dcase_.*'},
            {'icra_.*'},
            {'demand'},
            {'noisex'},
            {'arte'},
        ]
    )))


def noise_cross_corpus_fair():
    print('Noise cross-corpus fair')
    print(np.round(get_mean_gap_cross_corpus_fair(
        'noise_types',
        [
            {'dcase_.*'},
            {'icra_.*'},
            {'demand'},
            {'noisex'},
            {'arte'},
        ]
    )))


def surrey_naive():
    print('Surrey naive')
    print(np.round(get_mean_gap(
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
    )))


def ash_naive():
    print('ASH naive')
    print(np.round(get_mean_gap(
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
    )))


def ash_fair():
    print('ASH fair')
    print(np.round(get_mean_gap(
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
    )))


def room_cross_corpus_naive():
    print('Room cross-corpus naive')
    print(np.round(get_mean_gap_cross_corpus_naive(
        'rooms',
        [
            {'surrey_.*'},
            {'ash_.*'},
            {'air_.*'},
            {'catt_.*'},
            {'avil_.*'},
        ]
    )))


def room_cross_corpus_fair():
    print('Room cross-corpus fair')
    print(np.round(get_mean_gap_cross_corpus_fair(
        'rooms',
        [
            {'surrey_.*'},
            {'ash_.*'},
            {'air_.*'},
            {'catt_.*'},
            {'avil_.*'},
        ]
    )))


def rooms():
    print('rooms')
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
    print(np.round(gaps))


def snr():
    print('SNR')
    snr_dist_args = [
        [-5, -5],
        [0, 0],
        [5, 5],
        [10, 10],
        [-5, 10],
    ]
    gaps = np.zeros((3, 5, 5))
    for i, cond_dim_val in enumerate(snr_dist_args):
        for j, model_dim_val in enumerate(snr_dist_args):
            gaps[:, i, j] = get_generalization_gap(
                'snr_dist_args',
                model_dim_val,
                cond_dim_val,
            )
    print(np.round(gaps))


def direction():
    print('Direction')
    target_angle_lims = [
        [0.0, 0.0],
        [-90.0, 90.0],
    ]
    gaps = np.zeros((3, 2, 2))
    for i, cond_dim_val in enumerate(target_angle_lims):
        for j, model_dim_val in enumerate(target_angle_lims):
            gaps[:, i, j] = get_generalization_gap(
                'target_angle_lims',
                model_dim_val,
                cond_dim_val,
            )
    print(np.round(gaps))


def level():
    print('Level')
    rms_jitters = [
        False,
        True,
    ]
    gaps = np.zeros((3, 2, 2))
    for i, cond_dim_val in enumerate(rms_jitters):
        for j, model_dim_val in enumerate(rms_jitters):
            gaps[:, i, j] = get_generalization_gap(
                'random_rms',
                model_dim_val,
                cond_dim_val,
            )
    print(np.round(gaps))


# timit_naive()
# timit_fair()
# timit_wise()
# libri_naive()
# libri_fair()
# libri_wise()
# speaker_cross_corpus_naive()
# speaker_cross_corpus_fair()
# dcase_naive()
# # dcase_fair()
# noise_cross_corpus_naive()
# noise_cross_corpus_fair()
# # surrey_naive()
# ash_naive()
# ash_fair()
# room_cross_corpus_naive()
# room_cross_corpus_fair()

rooms()
snr()
direction()
level()
