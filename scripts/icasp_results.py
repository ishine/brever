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
            ref_model_dim_val,
            cond_dim_val,
        ):
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
        **{dim: ref_model_dim_val}
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


def timit_naive():
    print('TIMIT naive')
    gap = np.mean([
        get_generalization_gap(
            'speakers',
            speaker,
            {'timit_.*'},
            {'timit_.*'},
        )
        for speaker in [
            {'timit_m0'},  # male 0
            {'timit_f0'},  # female 0
            {'timit_m1'},  # male 1
            {'timit_f1'},  # female 1
            {'timit_m2'},  # male 2
            {'timit_f2'},  # female 2
        ]], axis=0)
    print(np.round(gap))


def timit_fair():
    print('TIMIT fair')
    gap = np.mean([
        get_generalization_gap(
            'speakers',
            speaker,
            {'timit_.*'},
            {'timit_.*'},
        )
        for speaker in [
            {'timit_(f[0-4]|m[0-4])'},  # males and females 0 to 4
            {'timit_(f[5-9]|m[5-9])'},  # males and females 5 to 9
            {'timit_(f1[0-4]|m1[0-4])'},  # males and females 10 to 14
            {'timit_(f1[5-9]|m1[5-9])'},  # males and females 15 to 19
            {'timit_(f2[0-4]|m2[0-4])'},  # males and females 20 to 24
        ]], axis=0)
    print(np.round(gap))


def timit_wise():
    print('TIMIT wise')
    gap = np.mean([
        get_generalization_gap(
            'speakers',
            speaker,
            {'timit_.*'},
            {'timit_.*'},
        )
        for speaker in [
            {'timit_(f[0-4]?[0-9]|m[0-4]?[0-9])'},  # males and females 0 to 49
            {'timit_(f[4-9][0-9]|m[4-9][0-9])'},  # males and females 49 to 99
            {'timit_(f1[0-4][0-9]|m1[0-4][0-9])'},  # males and females 100 to 149
            {'timit_(f[0-9]?[02468]|m[0-9]?[02468])'},  # even males and females 0 to 99
            {'timit_(f[0-9]?[13579]|m[0-9]?[13579])'},  # odd males and females 0 to 99
        ]], axis=0)
    print(np.round(gap))


def libri_naive():
    print('LibriSpeech naive')
    gap = np.mean([
        get_generalization_gap(
            'speakers',
            speaker,
            {'libri_.*'},
            {'libri_.*'},
        )
        for speaker in [
            {'libri_m0'},  # male 0
            {'libri_f0'},  # female 0
            {'libri_m1'},  # male 1
            {'libri_f1'},  # female 1
            {'libri_m2'},  # male 2
            {'libri_f2'},  # female 2
        ]], axis=0)
    print(np.round(gap))


def libri_fair():
    print('LibriSpeech fair')
    gap = np.mean([
        get_generalization_gap(
            'speakers',
            speaker,
            {'libri_.*'},
            {'libri_.*'},
        )
        for speaker in [
            {'libri_(f[0-4]|m[0-4])'},  # males and females 0 to 4
            {'libri_(f[5-9]|m[5-9])'},  # males and females 5 to 9
            {'libri_(f1[0-4]|m1[0-4])'},  # males and females 10 to 14
            {'libri_(f1[5-9]|m1[5-9])'},  # males and females 15 to 19
            {'libri_(f2[0-4]|m2[0-4])'},  # males and females 20 to 24
        ]], axis=0)
    print(np.round(gap))


def libri_wise():
    print('LibriSpeech wise')
    gap = np.mean([
        get_generalization_gap(
            'speakers',
            speaker,
            {'libri_.*'},
            {'libri_.*'},
        )
        for speaker in [
            {'libri_(f[0-4]?[0-9]|m[0-4]?[0-9])'},  # males and females 0 to 49
            {'libri_(f[4-9][0-9]|m[4-9][0-9])'},  # males and females 49 to 99
            {'libri_(f[0-9]?[02468]|m[0-9]?[02468])'},  # even males and females 0 to 99
            {'libri_(f[0-9]?[13579]|m[0-9]?[13579])'},  # odd males and females 0 to 99
        ]], axis=0)
    print(np.round(gap))


def speaker_cross_corpus_naive():
    print('Speaker cross-corpus naive')
    speakers = [
        {'ieee'},
        {'libri_.*'},
        {'timit_.*'},
        {'arctic'},
        {'hint'},
    ]
    ref_speakers = {'ieee', 'libri_.*', 'timit_.*', 'arctic', 'hint'}
    gaps = []
    for model_speaker in speakers:
        for cond_speaker in speakers:
            if cond_speaker != model_speaker:
                gaps.append(get_generalization_gap(
                    'speakers',
                    model_speaker,
                    ref_speakers,
                    cond_speaker,
                ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def speaker_cross_corpus_fair():
    print('Speaker cross-corpus naive')
    speakers = [
        {'ieee'},
        {'libri_.*'},
        {'timit_.*'},
        {'arctic'},
        {'hint'},
    ]
    ref_speakers = {'ieee', 'libri_.*', 'timit_.*', 'arctic', 'hint'}
    gaps = []
    for cond_speaker in speakers:
        gaps.append(get_generalization_gap(
            'speakers',
            model_speaker,
            ref_speakers,
            cond_speaker,
        ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def dcase_naive():
    print('DCASE naive')
    noises = [
        {'dcase_airport'},
        {'dcase_bus'},
        {'dcase_metro'},
        {'dcase_metro_station'},
        {'dcase_park'},
        {'dcase_public_square'},
        {'dcase_shopping_mall'},
        {'dcase_street_pedestrian'},
        {'dcase_street_traffic'},
        {'dcase_tram'},
    ]
    gaps = []
    for model_noise in noises:
        for cond_noise in noises:
            if cond_noise != model_noise:
                gaps.append(get_generalization_gap(
                    'noise_types',
                    model_noise,
                    cond_noise,
                ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def dcase_naive_2():
    print('DCASE naive 2')
    gap = np.mean([
        get_generalization_gap(
            'noise_types',
            noise,
            {'dcase_.*'},
        )
        for noise in [
            {'dcase_airport'},
            {'dcase_bus'},
            {'dcase_metro'},
            {'dcase_metro_station'},
            {'dcase_park'},
            {'dcase_public_square'},
            {'dcase_shopping_mall'},
            {'dcase_street_pedestrian'},
            {'dcase_street_traffic'},
            {'dcase_tram'},
        ]], axis=0)
    print(np.round(gap))


def dcase_fair_1():
    print('DCASE fair 1')
    model_noises = [
        {
            'dcase_airport',
            'dcase_bus',
            'dcase_metro',
            'dcase_metro_station',
            'dcase_park',
        },
        {
            'dcase_public_square',
            'dcase_shopping_mall',
            'dcase_street_pedestrian',
            'dcase_street_traffic',
            'dcase_tram',
        },
        {
            'dcase_airport',
            'dcase_metro',
            'dcase_park',
            'dcase_shopping_mall',
            'dcase_street_traffic',
        },
        {
            'dcase_bus',
            'dcase_metro_station',
            'dcase_public_square',
            'dcase_street_pedestrian',
            'dcase_tram',
        },
    ]
    cond_noises = [
        {'dcase_airport'},
        {'dcase_bus'},
        {'dcase_metro'},
        {'dcase_metro_station'},
        {'dcase_park'},
        {'dcase_public_square'},
        {'dcase_shopping_mall'},
        {'dcase_street_pedestrian'},
        {'dcase_street_traffic'},
        {'dcase_tram'},
    ]
    gaps = []
    for model_noise in model_noises:
        for cond_noise in cond_noises:
            if cond_noise.copy().pop() not in model_noise:
                gaps.append(get_generalization_gap(
                    'noise_types',
                    model_noise,
                    cond_noise,
                ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def dcase_fair_2():
    print('DCASE fair 2')
    gap = np.mean([
        get_generalization_gap(
            'noise_types',
            noise,
            {'dcase_.*'},
        )
        for noise in [
            {
                'dcase_airport',
                'dcase_bus',
                'dcase_metro',
                'dcase_metro_station',
                'dcase_park',
            },
            {
                'dcase_public_square',
                'dcase_shopping_mall',
                'dcase_street_pedestrian',
                'dcase_street_traffic',
                'dcase_tram',
            },
            {
                'dcase_airport',
                'dcase_metro',
                'dcase_park',
                'dcase_shopping_mall',
                'dcase_street_traffic',
            },
            {
                'dcase_bus',
                'dcase_metro_station',
                'dcase_public_square',
                'dcase_street_pedestrian',
                'dcase_tram',
            },
        ]], axis=0)
    print(np.round(gap))


def noise_cross_corpus_naive():
    print('Noise cross-corpus naive')
    noises = [
        {'dcase_.*'},
        {'icra_.*'},
        {'demand'},
        {'noisex'},
        {'arte'},
    ]
    gaps = []
    for model_noise in noises:
        for cond_noise in noises:
            if cond_noise != model_noise:
                gaps.append(get_generalization_gap(
                    'noise_types',
                    model_noise,
                    cond_noise,
                ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def noise_cross_corpus_fair():
    print('Noise cross-corpus fair')
    noises = [
        {'dcase_.*'},
        {'icra_.*'},
        {'demand'},
        {'noisex'},
        {'arte'},
    ]
    gaps = []
    for cond_noise in noises:
        model_noise = [s.copy() for s in noises if s != cond_noise]
        model_noise = set(s.pop() for s in model_noise)
        gaps.append(get_generalization_gap(
            'noise_types',
            model_noise,
            cond_noise,
        ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def surrey_naive_1():
    print('Surrey naive 1')
    rooms = [
        {'surrey_anechoic'},
        {'surrey_room_a'},
        {'surrey_room_b'},
        {'surrey_room_c'},
        {'surrey_room_d'},
    ]
    gaps = []
    for model_room in rooms:
        for cond_room in rooms:
            if cond_room != model_room:
                gaps.append(get_generalization_gap(
                    'rooms',
                    model_room,
                    cond_room,
                ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def surrey_naive_2():
    print('Surrey naive 2')
    gap = np.mean([
        get_generalization_gap(
            'rooms',
            room,
            {'surrey_.*'},
        )
        for room in [
            {'surrey_anechoic'},
            {'surrey_room_a'},
            {'surrey_room_b'},
            {'surrey_room_c'},
            {'surrey_room_d'},
        ]], axis=0)
    print(np.round(gap))


def ash_naive():
    print('ASH naive')
    gap = np.mean([
        get_generalization_gap(
            'rooms',
            room,
            {'ash_.*'},
        )
        for room in [
            {'ash_r01'},
            {'ash_r02'},
            {'ash_r03'},
            {'ash_r04'},
            {'ash_r05a?b?'},
        ]], axis=0)
    print(np.round(gap))


def ash_fair():
    print('ASH fair')
    gap = np.mean([
        get_generalization_gap(
            'rooms',
            room,
            {'ash_.*'},
        )
        for room in [
            {'ash_r0[0-9]a?b?'},  # 0 to 9
            {'ash_r1[0-9]'},  # 10 to 19
            {'ash_r2[0-9]'},  # 20 to 29
            {'ash_r3[0-9]'},  # 30 to 39
        ]], axis=0)
    print(np.round(gap))


def room_cross_corpus_naive():
    print('Room cross-corpus naive')
    rooms = [
        {'surrey_.*'},
        {'ash_.*'},
        {'air_.*'},
        {'catt_.*'},
        {'avil_.*'},
    ]
    gaps = []
    for model_room in rooms:
        for cond_room in rooms:
            if cond_room != model_room:
                gaps.append(get_generalization_gap(
                    'rooms',
                    model_room,
                    cond_room,
                ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def room_cross_corpus_fair():
    print('Room cross-corpus fair')
    rooms = [
        {'surrey_.*'},
        {'ash_.*'},
        {'air_.*'},
        {'catt_.*'},
        {'avil_.*'},
    ]
    gaps = []
    for cond_room in rooms:
        model_room = [s.copy() for s in rooms if s != cond_room]
        model_room = set(s.pop() for s in model_room)
        gaps.append(get_generalization_gap(
            'rooms',
            model_room,
            cond_room,
        ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def snr_naive_1():
    print('SNR naive 1')
    snrs = [
        [-5, -5],
        [0, 0],
        [5, 5],
        [10, 10],
    ]
    gaps = []
    for model_snr in snrs:
        for cond_snr in snrs:
            if cond_snr != model_snr:
                gaps.append(get_generalization_gap(
                    'snr_dist_args',
                    model_snr,
                    cond_snr,
                ))
    gap = np.mean(gaps, axis=0)
    print(np.round(gap))


def snr_naive_2():
    print('SNR naive 2')
    gap = np.mean([
        get_generalization_gap(
            'snr_dist_args',
            snr,
            [-5, 10],
        )
        for snr in [
            [-5, -5],
            [0, 0],
            [5, 5],
            [10, 10],
        ]], axis=0)
    print(np.round(gap))


def snr_worst_worst_case():
    print('SNR worst worst case')
    gap = get_generalization_gap(
        'snr_dist_args',
        [10, 10],
        [-5, -5],
    )
    print(np.round(gap))


def snr_best_worst_case():
    print('SNR best worst case')
    gap = get_generalization_gap(
        'snr_dist_args',
        [-5, -5],
        [10, 10],
    )
    print(np.round(gap))


def direction_naive():
    print('Direction naive')
    gap = get_generalization_gap(
        'target_angle_lims',
        [0, 0],
        [-90, -90],
    )
    print(np.round(gap))


def direction_fair():
    print('Direction fair')
    gap = get_generalization_gap(
        'target_angle_lims',
        [-90, -90],
        [0, 0],
    )
    print(np.round(gap))


def level_naive():
    print('Level naive')
    gap = get_generalization_gap(
        'random_rms',
        False,
        True,
    )
    print(np.round(gap))


def level_fair():
    print('Level fair')
    gap = get_generalization_gap(
        'random_rms',
        True,
        False,
    )
    print(np.round(gap))


timit_naive()
timit_fair()
timit_wise()
libri_naive()
libri_fair()
libri_wise()
speaker_cross_corpus_naive()
speaker_cross_corpus_fair()
dcase_naive_1()
dcase_naive_2()
dcase_fair_1()
dcase_fair_2()
noise_cross_corpus_naive()
noise_cross_corpus_fair()
surrey_naive_1()
surrey_naive_2()
ash_naive()
ash_fair()
room_cross_corpus_naive()
room_cross_corpus_fair()
snr_naive_1()
# snr_naive_2()
snr_worst_worst_case()
snr_best_worst_case()
# direction_naive()
# direction_fair()
# level_naive()
# level_fair()
