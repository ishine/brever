import os
import itertools

import brever.modelmanagement as bmm
from brever.config import defaults


def main(args, params):

    def_cfg = defaults()
    processed_dir = def_cfg.PATH.PROCESSED
    test_dir = os.path.join(processed_dir, 'test')

    new_configs = []
    new_paths = []
    old_configs = []
    old_paths = []

    default = {
        'room': {'surrey_.*'},
        'snr': [-5, 10],
        'noise': {'dcase_.*'},
        'speaker': {'timit_.*'},
        'rms': False,
        'angle': [0.0, 0.0],
    }

    for room, snr, noise, angle, rms, speaker in itertools.product(
                args.test_rooms,
                args.test_snrs,
                args.test_noises,
                args.test_angles,
                args.test_rms,
                args.test_speakers,
            ):

        if sum([
            room == default['room'],
            snr == default['snr'],
            noise == default['noise'],
            angle == default['angle'],
            rms == default['rms'],
            speaker == default['speaker'],
        ]) < 5:
            continue

        config = {
            'PRE': {
                'MIX': {
                    'TOTALDURATION': args.test_duration,
                    'FILELIMITS': {
                        'NOISE': [0.85, 1.0],
                        'TARGET': [0.85, 1.0]
                    },
                    'SAVE': True,
                    'RANDOM': {
                        'ROOMS': room,
                        'TARGET': {
                            'ANGLE': {
                                'MIN': angle[0],
                                'MAX': angle[1]
                            },
                            'SNR': {
                                'DISTARGS': [snr[0], snr[1]],
                                'DISTNAME': 'uniform'
                            },
                            'SPEAKERS': speaker
                        },
                        'SOURCES': {
                                'TYPES': noise
                        },
                        'RMSDB': {
                            'ON': rms
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

        dset_id = bmm.get_unique_id(config)
        dset_path = os.path.join(test_dir, dset_id)

        if not os.path.exists(dset_path):
            new_configs.append(config)
            new_paths.append(dset_path)
        else:
            old_configs.append(config)
            old_paths.append(dset_path)

    print(f'{len(new_paths) + len(old_paths)} datasets attempted to be '
          'initialized.')
    print(f'{len(old_paths)} already exist.')

    # build the list of dsets already in the filesystem
    filesystem_dsets = []
    for file in os.listdir(test_dir):
        filesystem_dsets.append(os.path.join(test_dir, file))
    # highlight the dsets in the filesystem that were not attempted to be
    # created again; they might be deprecated
    deprecated_dsets = []
    for dset in filesystem_dsets:
        if dset not in old_paths:
            deprecated_dsets.append(dset)
    if deprecated_dsets:
        print('The following datasets are in the filesystem but were not '
              'attempted to be initialized again. They might be deprecated?')
        for dset in deprecated_dsets:
            print(dset)

    if not new_paths:
        print(f'{len(new_paths)} will be initialized.')
    else:
        resp = input(f'{len(new_paths)} will be initialized. Continue? y/n')
        if resp == 'y':
            for config, dirpath in zip(new_configs, new_paths):
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
    parser = bmm.DatasetInitArgParser(description='initialize test datasets')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('--test-duration', type=int, default=300,
                        help='total duration, defaults to 300 seconds')
    parser.add_argument('--test-rooms', nargs='+',
                        type=bmm.arg_set_type,
                        default=[
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
                        ],
                        help='rooms for the grid of test conditions')
    parser.add_argument('--test-snrs', nargs='+',
                        type=lambda x: bmm.arg_list_type(x, int),
                        default=[
                            [0, 0],
                            [-5, -5],
                            [5, 5],
                            [10, 10],
                            [-5, 10],
                        ],
                        help='snr limits for the grid of test conditions')
    parser.add_argument('--test-noises', nargs='+',
                        type=bmm.arg_set_type,
                        default=[
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
                        ],
                        help='noises for the grid of test conditions')
    parser.add_argument('--test-angles', nargs='+',
                        type=lambda x: bmm.arg_list_type(x, int),
                        default=[
                            [0.0, 0.0],
                            [-90.0, 90.0],
                        ],
                        help='angle limits for the grid of test conditions')
    parser.add_argument('--test-rms',  nargs='+',
                        type=lambda x: bool(int(x)),
                        default=[
                            False,
                            True,
                        ],
                        help='random rms for the grid of test conditions')
    parser.add_argument('--test-speakers',
                        type=bmm.arg_set_type,
                        default=[
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
                        ],
                        help='speakers for the grid of test conditions')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
