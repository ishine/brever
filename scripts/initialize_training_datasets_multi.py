import os

import brever.modelmanagement as bmm
from brever.config import defaults


def main(args, params):

    def_cfg = defaults()
    processed_dir = def_cfg.PATH.PROCESSED

    # parameters for the general dataset
    general_noise_types = {
        'dcase_airport',
        'dcase_bus',
        'dcase_metro',
        'dcase_metro_station',
        'dcase_park',
        'dcase_public_square',
        'dcase_shopping_mall',
        'dcase_street_pedestrian',
        'dcase_street_traffic',
        'dcase_tram',
    }
    general_rooms = {
        'surrey_room_.',
        'ash_r.*',
    }
    general_angle_lims = (-90.0, 90.0)
    general_snr_lims = (-5, 10)
    general_rms_jitter = True
    general_speakers = {
        'timit_.*',
        'libri_*',
    }

    # actual grid of dataset parameters
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
        general_noise_types,
    ]
    roomss = [
        {'surrey_room_a'},
        {'surrey_room_b'},
        {'surrey_room_c'},
        {'surrey_room_d'},
        {'surrey_room_.'},
        {'ash_r.*'},
        {'ash_r01'},
        {'^ash_r(?!01$).*$'},
        {'ash_r0.*'},
        {'^ash_r(?!0).*$'},
        general_rooms,
    ]
    angle_limss = [
        (0.0, 0.0),
        general_angle_lims,
    ]
    snr_limss = [
        (-5, -5),
        (0, 0),
        (5, 5),
        (10, 10),
        general_snr_lims,
    ]
    rms_jitters = [
        False,
        general_rms_jitter,
    ]
    speakerss = [
        {'ieee'},
        {'timit_.*'},
        {'timit_FCJF0'},
        {'timit_^(?!FCJF0$).*$'},
        {'libri_.*'},
        {'libri_19'},
        {'libri_^(?!19$).*$'},
        general_speakers,
    ]

    # the default config is a very specialized one
    def add_config(
                configs,
                noise_types={'dcase_airport'},
                rooms={'surrey_room_a'},
                angle_lims=[0.0, 0.0],
                snr_lims=[0, 0],
                rms_jitter=False,
                speakers={'ieee'},
            ):

        is_libri = False
        for speaker in speakers:
            if speaker.startswith('libri'):
                is_libri = True
                break

        for basename, filelims, number, seed in [
                    ('train', [0.0, 0.7], args.n_train, args.seed_train),
                    ('val', [0.7, 0.85], args.n_val, args.seed_val),
                ]:

            if is_libri:
                number = number//10

            config = {
                'PRE': {
                    'SEED': {
                        'ON': True,
                        'VALUE': seed,
                    },
                    'MIX': {
                        'SAVE': False,
                        'NUMBER': number,
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

    # don't forget to add the general config!
    add_config(
        configs,
        noise_types=general_noise_types,
        rooms=general_rooms,
        angle_lims=general_angle_lims,
        snr_lims=general_snr_lims,
        rms_jitter=general_rms_jitter,
        speakers=general_speakers,
    )

    new_configs = []
    for config_dict, dset_path in configs:
        if not os.path.exists(dset_path):
            new_configs.append((config_dict, dset_path))

    print(f'{len(configs)} datasets attempted to be initialized.')
    print(f'{len(configs) - len(new_configs)} already exist.')
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
    parser.add_argument('--n-train', type=int, default=10000,
                        help='number of training mixture, defaults to 1000')
    parser.add_argument('--n-val', type=int, default=2000,
                        help='number of validation mixture, defaults to 200')
    parser.add_argument('--seed-train', type=int, default=0,
                        help='seed for the training dataset')
    parser.add_argument('--seed-val', type=int, default=1,
                        help='seed for the validation dataset')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
