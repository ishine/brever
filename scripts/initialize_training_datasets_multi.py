import os

import brever.modelmanagement as bmm
from brever.config import defaults


def main(args, params):

    def_cfg = defaults()
    processed_dir = def_cfg.PATH.PROCESSED

    # grid of dataset parameters
    noise_typess = [
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
        {'icra_01'},
        {'icra_02'},
        {'icra_03'},
        {'icra_04'},
        {'icra_05'},
        {'icra_06'},
        {'icra_07'},
        {'icra_08'},
        {'icra_09'},
        {'dcase_.*'},
        {'icra_.*'},
        {'demand'},
        {'noisex'},
        {'arte'},
        {'icra_.*', 'demand', 'noisex', 'arte'},
        {'dcase_.*', 'demand', 'noisex', 'arte'},
        {'dcase_.*', 'icra_.*', 'noisex', 'arte'},
        {'dcase_.*', 'icra_.*', 'demand', 'arte'},
        {'dcase_.*', 'icra_.*', 'demand', 'noisex'},
        {'dcase_.*', 'icra_.*', 'demand', 'noisex', 'arte'},
    ]
    roomss = [
        {'surrey_anechoic'},
        {'surrey_room_a'},
        {'surrey_room_b'},
        {'surrey_room_c'},
        {'surrey_room_d'},
        {'ash_r01'},
        {'ash_r02'},
        {'ash_r03'},
        {'ash_r04'},
        {'ash_r05a?b?'},
        {'ash_r0[0-9]a?b?'},  # 0 to 9
        {'ash_r1[0-9]'},  # 10 to 19
        {'ash_r2[0-9]'},  # 20 to 29
        {'ash_r3[0-9]'},  # 30 to 39
        {'ash_r[0-1][0248]a?b?'},  # even from 0 to 19
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
        # {'timit_(?!m0$).*'},  # male 0
        # {'timit_(?!f0$).*'},  # female 0
        # {'timit_(?!m1$).*'},  # male 1
        # {'timit_(?!f1$).*'},  # female 1
        # {'timit_(?!m2$).*'},  # male 2
        # {'timit_(?!f2$).*'},  # female 2
        # {'timit_(?!(f[0-4]|m[0-4])$).*'},  # males and females 0 to 4
        # {'timit_(?!(f[5-9]|m[5-9])$).*'},  # males and females 5 to 9
        # {'timit_(?!(f1[0-4]|m1[0-4])$).*'},  # males and females 10 to 14
        # {'timit_(?!(f1[5-9]|m1[5-9])$).*'},  # males and females 15 to 19
        # {'timit_(?!(f2[0-4]|m2[0-4])$).*'},  # males and females 20 to 24
        # {'timit_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$),*'},  # males and females 0 to 49
        # {'timit_(?!(f[4-9][0-9]|m[4-9][0-9])$),*'},  # males and females 49 to 99
        # {'timit_(?!(f1[0-4][0-9]|m1[0-4][0-9])$),*'},  # males and females 100 to 149
        # {'timit_(?!(f[0-9]?[02468]|m[0-9]?[02468])$),*'},  # even males and females 0 to 99
        # {'timit_(?!(f[0-9]?[13579]|m[0-9]?[13579])$),*'},  # odd males and females 0 to 99
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
        # {'libri_(?!m0$).*'},  # male 0
        # {'libri_(?!f0$).*'},  # female 0
        # {'libri_(?!m1$).*'},  # male 1
        # {'libri_(?!f1$).*'},  # female 1
        # {'libri_(?!m2$).*'},  # male 2
        # {'libri_(?!f2$).*'},  # female 2
        # {'libri_(?!(f[0-4]|m[0-4])$).*'},  # males and females 0 to 4
        # {'libri_(?!(f[5-9]|m[5-9])$).*'},  # males and females 5 to 9
        # {'libri_(?!(f1[0-4]|m1[0-4])$).*'},  # males and females 10 to 14
        # {'libri_(?!(f1[5-9]|m1[5-9])$).*'},  # males and females 15 to 19
        # {'libri_(?!(f2[0-4]|m2[0-4])$).*'},  # males and females 20 to 24
        # {'libri_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},  # males and females 0 to 49
        # {'libri_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},  # males and females 49 to 99
        # {'libri_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},  # even males and females 0 to 99
        # {'libri_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},  # odd males and females 0 to 99
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

    # the default config is a very specialized one
    def add_config(
                configs,
                noise_types={'dcase_.*'},
                rooms={'surrey_.*'},
                angle_lims=[0.0, 0.0],
                snr_lims=[-5, 10],
                rms_jitter=False,
                speakers={'timit_.*'},
            ):

        for basename, filelims, duration, seed in zip(
                    ('train', 'val'),
                    ([0.0, 0.7], [0.7, 0.85]),
                    (args.train_duration, args.val_duration),
                    (args.seed_train, args.seed_val),
                ):

            config = {
                'PRE': {
                    'SEED': {
                        'ON': True,
                        'VALUE': seed,
                    },
                    'MIX': {
                        'SAVE': False,
                        'TOTALDURATION': duration,
                        'FILELIMITS': {
                            'NOISE': filelims.copy(),
                            'TARGET': filelims.copy(),
                        },
                        'RANDOM': {
                            'ROOMS': rooms,
                            'TARGET': {
                                'ANGLE': {
                                    'MIN': angle_lims[0],
                                    'MAX': angle_lims[1]
                                },
                                'SNR': {
                                    'DISTARGS': [snr_lims[0], snr_lims[1]],
                                    'DISTNAME': 'uniform'
                                },
                                'SPEAKERS': speakers
                            },
                            'SOURCES': {
                                    'TYPES': noise_types
                            },
                            'RMSDB': {
                                'ON': rms_jitter
                            }
                        }
                    }
                }
            }

            for param, value in params.items():
                if value is not None:
                    key_list = bmm.DatasetInitArgParser.arg_to_keys_map[param]
                    bmm.set_dict_field(config, key_list, value)

            def_cfg.update(config)  # throws an error if config is not valid

            # use the same ID for both train and val datasets
            # namely use the ID obtained from the train config
            if basename == 'train':
                dset_id = bmm.get_unique_id(config)

            dset_path = os.path.join(processed_dir, basename, dset_id)

            if (config, dset_path) not in configs:
                configs.append((config, dset_path))

    configs = []

    # note that some configs below will be duplicate but the test at the end of
    # the function will correctly prevent from adding them
    for noise_types in noise_typess:
        add_config(configs, noise_types=noise_types)
    for rooms in roomss:
        add_config(configs, rooms=rooms)
    for angle_lims in angle_limss:
        add_config(configs, angle_lims=angle_lims)
    for snr_lims in snr_limss:
        add_config(configs, snr_lims=snr_lims)
    for rms_jitter in rms_jitters:
        add_config(configs, rms_jitter=rms_jitter)
    for speakers in speakerss:
        add_config(configs, speakers=speakers)

    new_configs = []
    for config_dict, dset_path in configs:
        if not os.path.exists(dset_path):
            new_configs.append((config_dict, dset_path))

    print(f'{len(configs)} datasets attempted to be initialized.')
    print(f'{len(configs) - len(new_configs)} already exist.')

    # build the list of dsets already in the filesystem
    filesystem_dsets = []
    for subdir in ['train', 'val']:
        for file in os.listdir(os.path.join(processed_dir, subdir)):
            filesystem_dsets.append(os.path.join(processed_dir, subdir, file))
    # highlight the dsets in the filesystem that were not attempted to be
    # created again; they might be deprecated
    deprecated_dsets = []
    for dset in filesystem_dsets:
        if dset not in [config[1] for config in configs]:
            deprecated_dsets.append(dset)
    if deprecated_dsets:
        print('The following datasets are in the filesystem but were not '
              'attempted to be initialized again. They might be deprecated?')
        for dset in deprecated_dsets:
            print(dset)

    if not new_configs:
        print(f'{len(new_configs)} will be initialized.')
    else:
        resp = input(f'{len(new_configs)} will be initialized. Continue? y/n')
        if resp == 'y':
            for config, dirpath in new_configs:
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                config_filepath = os.path.join(dirpath, 'config.yaml')
                if os.path.exists(config_filepath) and not args.force:
                    print(f'{config_filepath} already exists')
                    continue
                bmm.dump_yaml(config, config_filepath)
                print(f'Initialized {config_filepath}')
        else:
            print('No dataset was initialized.')


if __name__ == '__main__':
    parser = bmm.DatasetInitArgParser(description='initialize train datasets')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('--train-duration', type=int, default=36000,
                        help='training duration, defaults to 36000 seconds')
    parser.add_argument('--val-duration', type=int, default=7200,
                        help='validation duration, defaults to 7200 seconds')
    parser.add_argument('--seed-train', type=int, default=0,
                        help='seed for the training dataset')
    parser.add_argument('--seed-val', type=int, default=1,
                        help='seed for the validation dataset')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
