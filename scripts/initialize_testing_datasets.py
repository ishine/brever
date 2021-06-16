import os
import itertools

import yaml

import brever.modelmanagement as bmm
from brever.config import defaults


def main(args, params):

    processed_dir = defaults().PATH.PROCESSED

    new_configs = []
    new_paths = []
    old_configs = []
    old_paths = []

    for room, snr, noise, angle, rms in itertools.product(
                args.test_rooms,
                args.test_snrs,
                args.test_noises,
                args.test_angles,
                args.test_rms,
            ):
        config = {
            'PRE': {
                'MIX': {
                    'NUMBER': args.n_test,
                    'FILELIMITS': {
                        'NOISE': [0.85, 1.0],
                        'TARGET': [0.85, 1.0]
                    },
                    'SAVE': True,
                    'RANDOM': {
                        'ROOMS': room,
                        'TARGET': {
                            'ANGLE': {
                                'MAX': angle[0],
                                'MIN': angle[1]
                            },
                            'SNR': {
                                'DISTARGS': [snr[0], snr[1]],
                                'DISTNAME': 'uniform'
                            }
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

        defaults().update(config)  # throws an error if config is not valid

        dset_id = bmm.get_unique_id(config)
        dset_name = f'autoinit_{dset_id[:6]}'
        test_dir = os.path.join(processed_dir, 'test')
        dset_path = os.path.join(test_dir, dset_name)

        if not os.path.exists(dset_path):
            new_configs.append(config)
            new_paths.append(dset_path)
        else:
            old_configs.append(config)
            old_paths.append(dset_path)

    print(f'{len(new_paths) + len(old_paths)} datasets attempted to be '
          'initialized.')
    print(f'{len(old_paths)} already exist.')

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
                with open(config_filepath, 'w') as f:
                    yaml.dump(config, f)
                print(f'Initialized {config_filepath}')
        else:
            print('No dataset was initialized.')


if __name__ == '__main__':
    parser = bmm.DatasetInitArgParser(description='initialize test datasets')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('--n-test', type=int, default=10,
                        help='number of test mixtures, defaults to 10')
    parser.add_argument('--test-rooms', nargs='+',
                        type=bmm.arg_set_type,
                        default=[
                            {'surrey_room_a'},
                            {'surrey_room_b'},
                            {'surrey_room_c'},
                            {'surrey_room_d'},
                        ],
                        help='rooms for the grid of test conditions')
    parser.add_argument('--test-snrs', nargs='+',
                        type=lambda x: bmm.arg_list_type(x, int),
                        default=[
                            [-5, 5],
                            [0, 0],
                            [5, 5],
                            [10, 10],
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
                            {'dcase_public_square'},
                            {'dcase_shopping_mall'},
                            {'dcase_street_pedestrian'},
                            {'dcase_street_traffic'},
                            {'dcase_tram'},
                        ],
                        help='noises for the grid of test conditions')
    parser.add_argument('--test-angles', nargs='+',
                        type=lambda x: bmm.arg_list_type(x, int),
                        default=[
                            [0.0, 0.0],
                            [-90.0, 90.0],
                        ],
                        help='angle limits for the grid of test conditions')
    parser.add_argument('--test-rms', type=lambda x: bool(int(x)), nargs='+',
                        default=[
                            False,
                            True,
                        ],
                        help='random rms for the grid of test conditions')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
