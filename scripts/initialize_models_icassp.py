import os
import shutil
import itertools
import copy

from brever.config import defaults
import brever.modelmanagement as bmm


def check_if_path_exists(configs, path_type, def_cfg):
    if path_type not in ['train', 'val']:
        raise ValueError('path_type must be train or val')
    default_path = bmm.get_config_field(def_cfg, f'{path_type}_path')
    for config in configs:
        path = bmm.get_config_field(config, f'{path_type}_path')
        if path is None:
            path = default_path
            msg = f'No {path_type} path specified, and default path does not '\
                  'exist'
        else:
            msg = f'The specified {path_type} path does not exist'
        if not os.path.exists(path):
            print(msg)
            return ask_user_yes_no('Do you wish to continue? y/n')
    return True


def check_if_test_datasets_exist(configs, def_cfg):
    default_paths = bmm.get_config_field(def_cfg, 'test_path')
    for config in configs:
        paths = bmm.get_config_field(config, 'test_path')
        if paths is None:
            paths = default_paths
            msg = 'No test paths specified, and not all the default test '\
                  'paths exist'
        else:
            msg = 'The specified test paths do not all exist'
        paths = bmm.globbed(paths)
        if not paths or any(not os.path.exists(path) for path in paths):
            print(msg)
            return ask_user_yes_no('Do you wish to continue? y/n')
    return True


def check_paths(configs, def_cfg):
    return (
        check_if_path_exists(configs, 'train', def_cfg)
        and check_if_path_exists(configs, 'val', def_cfg)
        and check_if_test_datasets_exist(configs, def_cfg)
    )


def ask_user_yes_no(msg):
    resp = None
    while resp not in ['y', 'Y', 'n', 'N']:
        resp = input(msg)
        if resp in ['y', 'Y']:
            return True
        elif resp in ['n', 'N']:
            return False
        else:
            print('Could not interpret answer')


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


def main(args):

    train_dsets, train_configs = bmm.find_dataset('train', return_configs=True)
    test_dsets, test_configs = bmm.find_dataset('test', return_configs=True)

    # ADD SPEECH, NOISE AND ROOM MODELS

    speakerss = [
        {'timit_m0'},  # male 0
        {'timit_f0'},  # female 0
        {'timit_m1'},  # male 1
        {'timit_f1'},  # female 1
        {'timit_m2'},  # male 2
        {'timit_f2'},  # female 2
        {'timit_(f[0-4]|m[0-4])'},  # males and females 0 to 4
        {'timit_(f[5-9]|m[5-9])'},  # males and females 5 to 9
        {'timit_(f1[0-4]|m1[0-4])'},  # males and females 10 to 14
        {'timit_(f1[5-9]|m1[5-9])'},  # males and females 15 to 19
        {'timit_(f2[0-4]|m2[0-4])'},  # males and females 20 to 24
        {'timit_(f[0-4]?[0-9]|m[0-4]?[0-9])'},  # males and females 0 to 49
        {'timit_(f[4-9][0-9]|m[4-9][0-9])'},  # males and females 49 to 99
        {'timit_(f1[0-4][0-9]|m1[0-4][0-9])'},  # males and females 100 to 149
        {'timit_(f[0-9]?[02468]|m[0-9]?[02468])'},  # even males and females 0 to 99
        {'timit_(f[0-9]?[13579]|m[0-9]?[13579])'},  # odd males and females 0 to 99
        {'timit_(?!m0$).*'},  # male 0
        {'timit_(?!f0$).*'},  # female 0
        {'timit_(?!m1$).*'},  # male 1
        {'timit_(?!f1$).*'},  # female 1
        {'timit_(?!m2$).*'},  # male 2
        {'timit_(?!f2$).*'},  # female 2
        {'timit_(?!(f[0-4]|m[0-4])$).*'},  # males and females 0 to 4
        {'timit_(?!(f[5-9]|m[5-9])$).*'},  # males and females 5 to 9
        {'timit_(?!(f1[0-4]|m1[0-4])$).*'},  # males and females 10 to 14
        {'timit_(?!(f1[5-9]|m1[5-9])$).*'},  # males and females 15 to 19
        {'timit_(?!(f2[0-4]|m2[0-4])$).*'},  # males and females 20 to 24
        {'timit_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},  # males and females 0 to 49
        {'timit_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},  # males and females 49 to 99
        {'timit_(?!(f1[0-4][0-9]|m1[0-4][0-9])$).*'},  # males and females 100 to 149
        {'timit_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},  # even males and females 0 to 99
        {'timit_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},  # odd males and females 0 to 99
        {'libri_m0'},  # male 0
        {'libri_f0'},  # female 0
        {'libri_m1'},  # male 1
        {'libri_f1'},  # female 1
        {'libri_m2'},  # male 2
        {'libri_f2'},  # female 2
        {'libri_(f[0-4]|m[0-4])'},  # males and females 0 to 4
        {'libri_(f[5-9]|m[5-9])'},  # males and females 5 to 9
        {'libri_(f1[0-4]|m1[0-4])'},  # males and females 10 to 14
        {'libri_(f1[5-9]|m1[5-9])'},  # males and females 15 to 19
        {'libri_(f2[0-4]|m2[0-4])'},  # males and females 20 to 24
        {'libri_(f[0-4]?[0-9]|m[0-4]?[0-9])'},  # males and females 0 to 49
        {'libri_(f[4-9][0-9]|m[4-9][0-9])'},  # males and females 49 to 99
        {'libri_(f[0-9]?[02468]|m[0-9]?[02468])'},  # even males and females 0 to 99
        {'libri_(f[0-9]?[13579]|m[0-9]?[13579])'},  # odd males and females 0 to 99
        {'libri_(?!m0$).*'},  # male 0
        {'libri_(?!f0$).*'},  # female 0
        {'libri_(?!m1$).*'},  # male 1
        {'libri_(?!f1$).*'},  # female 1
        {'libri_(?!m2$).*'},  # male 2
        {'libri_(?!f2$).*'},  # female 2
        {'libri_(?!(f[0-4]|m[0-4])$).*'},  # males and females 0 to 4
        {'libri_(?!(f[5-9]|m[5-9])$).*'},  # males and females 5 to 9
        {'libri_(?!(f1[0-4]|m1[0-4])$).*'},  # males and females 10 to 14
        {'libri_(?!(f1[5-9]|m1[5-9])$).*'},  # males and females 15 to 19
        {'libri_(?!(f2[0-4]|m2[0-4])$).*'},  # males and females 20 to 24
        {'libri_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},  # males and females 0 to 49
        {'libri_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},  # males and females 49 to 99
        {'libri_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},  # even males and females 0 to 99
        {'libri_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},  # odd males and females 0 to 99
        {'ieee'},
        {'timit_.*'},
        {'libri_.*'},
        {'arctic'},
        {'hint'},
        {'libri_.*', 'timit_.*', 'arctic', 'hint'},
        {'ieee', 'timit_.*', 'arctic', 'hint'},
        {'ieee', 'libri_.*', 'arctic', 'hint'},
        {'ieee', 'libri_.*', 'timit_.*', 'hint'},
        {'ieee', 'libri_.*', 'timit_.*', 'arctic'},
        {'ieee', 'libri_.*', 'timit_.*', 'arctic', 'hint'},
    ]
    roomss = [
        {'surrey_anechoic'},
        {'surrey_room_a'},
        {'surrey_room_b'},
        {'surrey_room_c'},
        {'surrey_room_d'},
        {'surrey_(?!anechoic$).*'},
        {'surrey_(?!room_a$).*'},
        {'surrey_(?!room_b$).*'},
        {'surrey_(?!room_c$).*'},
        {'surrey_(?!room_d$).*'},
        {'ash_r01'},
        {'ash_r02'},
        {'ash_r03'},
        {'ash_r04'},
        {'ash_r05a?b?'},
        {'ash_(?!r01$).*'},
        {'ash_(?!r02$).*'},
        {'ash_(?!r03$).*'},
        {'ash_(?!r04$).*'},
        {'ash_(?!r05a?b?$).*'},
        {'ash_r0[0-9]a?b?'},  # 0 to 9
        {'ash_r1[0-9]'},  # 10 to 19
        {'ash_r2[0-9]'},  # 20 to 29
        {'ash_r3[0-9]'},  # 30 to 39
        {'ash_r(00|04|08|12|16|20|24|18|32|36)'},  # every 4th room from 0 to 39
        {'ash_(?!r0[0-9]a?b?$).*'},  # 0 to 9
        {'ash_(?!r1[0-9]$).*'},  # 10 to 19
        {'ash_(?!r2[0-9]$).*'},  # 20 to 29
        {'ash_(?!r3[0-9]$).*'},  # 30 to 39
        {'ash_(?!r(00|04|08|12|16|20|24|18|32|36)$).*'},  # every 4th room from 0 to 39
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
        {'surrey_.*', 'ash_.*', 'air_.*', 'catt_.*', 'avil_.*'},
    ]
    noise_typess = [
        {'dcase_airport'},
        {'dcase_bus'},
        {'dcase_metro'},
        {'dcase_metro_station'},
        {'dcase_park'},
        # {'dcase_public_square'},
        # {'dcase_shopping_mall'},
        # {'dcase_street_pedestrian'},
        # {'dcase_street_traffic'},
        # {'dcase_tram'},
        {'dcase_(?!airport$).*'},
        {'dcase_(?!bus$).*'},
        {'dcase_(?!metro$).*'},
        {'dcase_(?!metro_station$).*'},
        {'dcase_(?!park$).*'},
        # {'dcase_(?!public_square$).*'},
        # {'dcase_(?!shopping_mall$).*'},
        # {'dcase_(?!street_pedestrian$).*'},
        # {'dcase_(?!street_traffic$).*'},
        # {'dcase_(?!tram$).*'},
        {'noisex_babble'},
        {'noisex_buccaneer1'},
        # {'noisex_buccaneer2'},
        {'noisex_destroyerengine'},
        # {'noisex_destroyerops'},
        {'noisex_f16'},
        {'noisex_factory1'},
        # {'noisex_factory2'},
        # {'noisex_hfchannel'},
        # {'noisex_leopard'},
        # {'noisex_m109'},
        # {'noisex_machinegun'},
        # {'noisex_pink'},
        # {'noisex_volvo'},
        # {'noisex_white'},
        {'noisex_(?!babble$).*'},
        {'noisex_(?!buccaneer1$).*'},
        # {'noisex_(?!buccaneer2$).*'},
        {'noisex_(?!destroyerengine$).*'},
        # {'noisex_(?!destroyerops$).*'},
        {'noisex_(?!f16$).*'},
        {'noisex_(?!factory1$).*'},
        # {'noisex_(?!factory2$).*'},
        # {'noisex_(?!hfchannel$).*'},
        # {'noisex_(?!leopard$).*'},
        # {'noisex_(?!m109$).*'},
        # {'noisex_(?!machinegun$).*'},
        # {'noisex_(?!pink$).*'},
        # {'noisex_(?!volvo$).*'},
        # {'noisex_(?!white$).*'},
        {'dcase_.*'},
        {'icra_.*'},
        {'noisex_.*'},
        {'demand'},
        {'arte'},
        {'icra_.*', 'demand', 'noisex_.*', 'arte'},
        {'dcase_.*', 'demand', 'noisex_.*', 'arte'},
        {'dcase_.*', 'icra_.*', 'noisex_.*', 'arte'},
        {'dcase_.*', 'icra_.*', 'demand', 'arte'},
        {'dcase_.*', 'icra_.*', 'demand', 'noisex_.*'},
        {'dcase_.*', 'icra_.*', 'demand', 'noisex_.*', 'arte'},
    ]

    test_speakers = [
        {'timit_(?!m0$).*'},  # male 0
        {'timit_(?!f0$).*'},  # female 0
        {'timit_(?!m1$).*'},  # male 1
        {'timit_(?!f1$).*'},  # female 1
        {'timit_(?!m2$).*'},  # male 2
        {'timit_(?!f2$).*'},  # female 2
        {'timit_(?!(f[0-4]|m[0-4])$).*'},  # males and females 0 to 4
        {'timit_(?!(f[5-9]|m[5-9])$).*'},  # males and females 5 to 9
        {'timit_(?!(f1[0-4]|m1[0-4])$).*'},  # males and females 10 to 14
        {'timit_(?!(f1[5-9]|m1[5-9])$).*'},  # males and females 15 to 19
        {'timit_(?!(f2[0-4]|m2[0-4])$).*'},  # males and females 20 to 24
        {'timit_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},  # males and females 0 to 49
        {'timit_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},  # males and females 49 to 99
        {'timit_(?!(f1[0-4][0-9]|m1[0-4][0-9])$).*'},  # males and females 100 to 149
        {'timit_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},  # even males and females 0 to 99
        {'timit_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},  # odd males and females 0 to 99
        {'libri_(?!m0$).*'},  # male 0
        {'libri_(?!f0$).*'},  # female 0
        {'libri_(?!m1$).*'},  # male 1
        {'libri_(?!f1$).*'},  # female 1
        {'libri_(?!m2$).*'},  # male 2
        {'libri_(?!f2$).*'},  # female 2
        {'libri_(?!(f[0-4]|m[0-4])$).*'},  # males and females 0 to 4
        {'libri_(?!(f[5-9]|m[5-9])$).*'},  # males and females 5 to 9
        {'libri_(?!(f1[0-4]|m1[0-4])$).*'},  # males and females 10 to 14
        {'libri_(?!(f1[5-9]|m1[5-9])$).*'},  # males and females 15 to 19
        {'libri_(?!(f2[0-4]|m2[0-4])$).*'},  # males and females 20 to 24
        {'libri_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},  # males and females 0 to 49
        {'libri_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},  # males and females 49 to 99
        {'libri_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},  # even males and females 0 to 99
        {'libri_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},  # odd males and females 0 to 99
        {'ieee'},
        {'timit_.*'},
        {'libri_.*'},
        {'arctic'},
        {'hint'},
    ]

    test_rooms = [
        {'surrey_anechoic'},
        {'surrey_room_a'},
        {'surrey_room_b'},
        {'surrey_room_c'},
        {'surrey_room_d'},
        {'surrey_(?!anechoic$).*'},
        {'surrey_(?!room_a$).*'},
        {'surrey_(?!room_b$).*'},
        {'surrey_(?!room_c$).*'},
        {'surrey_(?!room_d$).*'},
        {'ash_(?!r01$).*'},
        {'ash_(?!r02$).*'},
        {'ash_(?!r03$).*'},
        {'ash_(?!r04$).*'},
        {'ash_(?!r05a?b?$).*'},
        {'ash_(?!r0[0-9]a?b?$).*'},  # 0 to 9
        {'ash_(?!r1[0-9]$).*'},  # 10 to 19
        {'ash_(?!r2[0-9]$).*'},  # 20 to 29
        {'ash_(?!r3[0-9]$).*'},  # 30 to 39
        {'ash_(?!r(00|04|08|12|16|20|24|18|32|36)$).*'},  # every 4th room from 0 to 39
        {'surrey_.*'},
        {'ash_.*'},
        {'air_.*'},
        {'catt_.*'},
        {'avil_.*'},
    ]

    test_noise_typess = [
        {'dcase_airport'},
        {'dcase_bus'},
        {'dcase_metro'},
        {'dcase_metro_station'},
        {'dcase_park'},
        {'dcase_(?!airport$).*'},
        {'dcase_(?!bus$).*'},
        {'dcase_(?!metro$).*'},
        {'dcase_(?!metro_station$).*'},
        {'dcase_(?!park$).*'},
        {'noisex_babble'},
        {'noisex_buccaneer1'},
        {'noisex_destroyerengine'},
        {'noisex_f16'},
        {'noisex_factory1'},
        {'noisex_(?!babble$).*'},
        {'noisex_(?!buccaneer1$).*'},
        {'noisex_(?!destroyerengine$).*'},
        {'noisex_(?!f16$).*'},
        {'noisex_(?!factory1$).*'},
        {'dcase_.*'},
        {'icra_.*'},
        {'noisex_.*'},
        {'demand'},
        {'arte'},
    ]

    train_paths = []
    val_paths = []
    test_paths = []
    for dim, vect, test_vect in zip(
                ['speakers', 'rooms', 'noise_types'],
                [speakerss, roomss, noise_typess],
                [test_speakers, test_rooms, test_noise_typess],
            ):
        test_dset = set()
        for x in test_vect:
            dset, = find_dset(
                dsets=test_dsets,
                configs=test_configs,
                **{dim: x},
            ) 
            test_dset.add(dset)
        for x in vect:
            dset, = find_dset(
                dsets=train_dsets,
                configs=train_configs,
                **{dim: x},
            )
            if dset not in train_paths:
                train_paths.append(dset)
                dset = dset.replace('train', 'val')
                val_paths.append(dset)
                test_paths.append(test_dset)
            else:
                i = train_paths.index(dset)
                test_paths[i] = test_paths[i] | test_dset

    args.train_path = train_paths
    args.val_path = val_paths
    args.test_path = test_paths
    args.layers = [2]
    args.hidden_sizes = [[1024, 1024]]
    args.stacks = [5]
    args.dropout = [True]
    args.seed = [0]

    # # ask if all train and val path combinations should be done
    # combine_paths = True
    # if args.train_path is not None and args.val_path is not None:
    #     n_train = len(args.train_path)
    #     n_val = len(args.val_path)
    #     if n_train == n_val and n_train != 1:
    #         n_combis = n_train*n_val
    #         msg = ('Same number of train and val paths detected. Do you wish '
    #                'to initialize all pair combinations, or only as many '
    #                f'pairs as paths specified? [Y to do all {n_combis} '
    #                f'combinations / N to do only {n_train} pairs]')
    #         combine_paths = ask_user_yes_no(msg)
    n_train = len(args.train_path)
    combine_paths = False

    # create dict of parameters to combine
    to_combine = {}
    for key in bmm.ModelFilterArgParser.arg_to_keys_map.keys():
        if key in ['train_path', 'val_path', 'test_path'] and not combine_paths:
            continue
        value = args.__getattribute__(key)
        if value is not None:
            bmm.set_config_field(to_combine, key, value)
    if not combine_paths:
        bmm.set_dict_field(to_combine, ['path_index'], list(range(n_train)))

    # make combinations
    if to_combine:
        to_combine = bmm.flatten(to_combine)
        keys, values = zip(*to_combine.items())
        configs = bmm.unflatten(keys, itertools.product(*values))
    else:
        configs = [{}]

    # add train and val paths if user requested to not make all combinations
    if not combine_paths:
        for config in configs:
            path_index = config['path_index']
            train_path = args.train_path[path_index]
            val_path = args.val_path[path_index]
            test_path = args.test_path[path_index]
            bmm.set_config_field(config, 'train_path', train_path)
            bmm.set_config_field(config, 'val_path', val_path)
            bmm.set_config_field(config, 'test_path', test_path)
            config.pop('path_index')

    all_configs = configs

    # ADD SNR, DIRECTION AND LEVEL MODELS

    angle_limss = [
        [0.0, 0.0],
        [-90.0, 90.0],
    ]
    snr_limss = [
        [-5, -5],
        [0, 0],
        [5, 5],
        [10, 10],
        [-5, 10],
    ]
    rms_jitters = [
        False,
        True,
    ]

    train_paths = []
    val_paths = []
    test_paths = []
    for dim, vect in zip(
                ['target_angle_lims', 'snr_dist_args', 'random_rms'],
                [angle_limss, snr_limss, rms_jitters],
            ):
        test_dset = set()
        for x in vect:
            dset, = find_dset(
                dsets=test_dsets,
                configs=test_configs,
                **{dim: x},
            ) 
            test_dset.add(dset)
        for x in vect:
            dset, = find_dset(
                dsets=train_dsets,
                configs=train_configs,
                **{dim: x},
            )
            if dset not in train_paths:
                train_paths.append(dset)
                dset = dset.replace('train', 'val')
                val_paths.append(dset)
                test_paths.append(test_dset)
            else:
                i = train_paths.index(dset)
                test_paths[i] = test_paths[i] | test_dset

    args.train_path = train_paths
    args.val_path = val_paths
    args.test_path = test_paths
    args.layers = [2]
    args.hidden_sizes = [[1024, 1024]]
    args.stacks = [5]
    args.dropout = [True]
    args.seed = [0, 1, 2, 3, 4]

    # # ask if all train and val path combinations should be done
    # combine_paths = True
    # if args.train_path is not None and args.val_path is not None:
    #     n_train = len(args.train_path)
    #     n_val = len(args.val_path)
    #     if n_train == n_val and n_train != 1:
    #         n_combis = n_train*n_val
    #         msg = ('Same number of train and val paths detected. Do you wish '
    #                'to initialize all pair combinations, or only as many '
    #                f'pairs as paths specified? [Y to do all {n_combis} '
    #                f'combinations / N to do only {n_train} pairs]')
    #         combine_paths = ask_user_yes_no(msg)
    n_train = len(args.train_path)
    combine_paths = False

    # create dict of parameters to combine
    to_combine = {}
    for key in bmm.ModelFilterArgParser.arg_to_keys_map.keys():
        if key in ['train_path', 'val_path', 'test_path'] and not combine_paths:
            continue
        value = args.__getattribute__(key)
        if value is not None:
            bmm.set_config_field(to_combine, key, value)
    if not combine_paths:
        bmm.set_dict_field(to_combine, ['path_index'], list(range(n_train)))

    # make combinations
    if to_combine:
        to_combine = bmm.flatten(to_combine)
        keys, values = zip(*to_combine.items())
        configs = bmm.unflatten(keys, itertools.product(*values))
    else:
        configs = [{}]

    # add train and val paths if user requested to not make all combinations
    if not combine_paths:
        for config in configs:
            path_index = config['path_index']
            train_path = args.train_path[path_index]
            val_path = args.val_path[path_index]
            test_path = args.test_path[path_index]
            bmm.set_config_field(config, 'train_path', train_path)
            bmm.set_config_field(config, 'val_path', val_path)
            bmm.set_config_field(config, 'test_path', test_path)
            config.pop('path_index')

    all_configs += configs

    doubles = []
    for i, config_1 in enumerate(all_configs):
        for j, config_2 in enumerate(all_configs):
            if i != j:
                train_1 = bmm.get_config_field(config_1, 'train_path')
                train_2 = bmm.get_config_field(config_2, 'train_path')
                seed_1 = bmm.get_config_field(config_1, 'seed')
                seed_2 = bmm.get_config_field(config_2, 'seed')
                if train_1 == train_2 and seed_1 == seed_2:
                    doubles.append((i, j))
    assert len(doubles) == 2
    assert doubles[0][0] == doubles[1][1]
    assert doubles[0][1] == doubles[1][0]

    merged_paths = bmm.get_config_field(
        all_configs[doubles[0][0]],
        'test_path',
    ) | bmm.get_config_field(
        all_configs[doubles[0][1]],
        'test_path',
    )

    bmm.set_config_field(
        all_configs[doubles[0][0]],
        'test_path',
        merged_paths,
    )
    bmm.set_config_field(
        all_configs[doubles[0][1]],
        'test_path',
        merged_paths,
    )

    configs = all_configs

    def_cfg = defaults()
    models_dir = def_cfg.PATH.MODELS

    # check if paths exist
    result = check_paths(configs, def_cfg.to_dict())
    if not result:
        print('Aborting')
        return

    new_configs = []
    skipped = 0
    dupes = 0
    exists = 0

    for config in configs:
        def_cfg.update(config)  # throws an error if config is not valid

        model_id = bmm.get_unique_id(config)
        model_dir = os.path.join(models_dir, model_id)

        if os.path.exists(model_dir):
            exists += 1
            continue

        # exclude configs with uniform normalization features not included
        # in the list of features
        uni_feats = bmm.get_config_field(config, 'uni_norm_features', None)
        features = bmm.get_config_field(config, 'features', None)
        if (uni_feats is not None and features is not None
                and not uni_feats.issubset(features)):
            skipped += 1
            continue

        if config not in new_configs:
            new_configs.append(config)
        else:
            dupes += 1

    totally_new_configs = []
    exist_but_with_different_test_path = []
    existing_models = []
    existing_configs = []
    for model_id in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_id)
        cfg_path = os.path.join(model_dir, 'config.yaml')
        cfg = bmm.read_yaml(cfg_path)
        existing_models.append(model_dir)
        existing_configs.append(cfg)
    existing_tests = [c['POST']['PATH'].pop('TEST') for c in existing_configs]
    for config in new_configs:
        model_id = bmm.get_unique_id(config)
        model_dir = os.path.join(models_dir, model_id)
        if not(os.path.exists(model_dir)):
            copy_ = copy.deepcopy(config)   
            copy_id = bmm.get_unique_id(config)
            new_tests = copy_['POST']['PATH'].pop('TEST')
            try:
                index = existing_configs.index(copy_)
            except ValueError:
                totally_new_configs.append(config)
            else:
                exist_but_with_different_test_path.append((
                    os.path.join(models_dir, copy_id),
                    new_tests,
                    existing_models[index],
                    existing_tests[index]
                ))

    if exist_but_with_different_test_path:
        print(f'{len(exist_but_with_different_test_path)} models were '
              'attempted to be initialized but already exist using different '
              'test paths')
        msg = 'Would you like to add the test paths to the old list of ' \
              'paths intead? [overwrite/merge/new]'
        resp = None
        while resp not in ['overwrite', 'merge', 'new']:
            resp = input(msg)
            if resp.lower() == 'overwrite':
                for model, tests, old_model, old_tests in exist_but_with_different_test_path:
                    cfg_path = os.path.join(old_model, 'config.yaml')
                    cfg = bmm.read_yaml(cfg_path)
                    bmm.set_config_field(cfg, 'test_path', tests)
                    bmm.dump_yaml(cfg, cfg_path)
                new_configs = totally_new_configs
                break
            elif resp.lower() == 'merge':
                for model, tests, old_model, old_tests in exist_but_with_different_test_path:
                    cfg_path = os.path.join(old_model, 'config.yaml')
                    cfg = bmm.read_yaml(cfg_path)
                    bmm.set_config_field(cfg, 'test_path', tests | old_tests)
                    bmm.dump_yaml(cfg, cfg_path)
                new_configs = totally_new_configs
                break
            elif resp.lower() == 'new':
                break
            else:
                print('Could not interpret answer')

    print(f'{len(configs)-skipped} config(s) attempted to be initialized.')
    print(f'{exists} already exist.')
    print(f'{dupes} are duplicates.')

    if not new_configs:
        print(f'{len(new_configs)} will be initialized.')
    else:
        msg = f'{len(new_configs)} will be initialized. Continue? y/n'
        proceed = ask_user_yes_no(msg)
        if proceed:
            for config in new_configs:
                unique_id = bmm.get_unique_id(config)
                dirpath = os.path.join(models_dir, unique_id)
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
                os.makedirs(dirpath)
                bmm.dump_yaml(config, os.path.join(dirpath, 'config.yaml'))
                print(f'Initialized {unique_id}')
        else:
            print('No model was initialized')


if __name__ == '__main__':
    parser = bmm.ModelFilterArgParser(description='initialize models')
    args, _ = parser.parse_args()
    main(args)
