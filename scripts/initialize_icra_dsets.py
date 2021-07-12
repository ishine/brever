import os
import itertools

import brever.modelmanagement as bmm
from brever.config import defaults


def main(args, params):

    def_cfg = defaults()
    processed_dir = def_cfg.PATH.PROCESSED

    new_configs = []
    new_paths = []
    old_configs = []
    old_paths = []

    for room, snr, noise, angle, rms, speaker in itertools.product(
                args.test_rooms,
                args.test_snrs,
                args.test_noises,
                args.test_angles,
                args.test_rms,
                args.test_speakers,
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

        dset_id = ''.join(sorted(noise))
        test_dir = os.path.join(processed_dir, 'test')
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
    parser.add_argument('--n-test', type=int, default=100,
                        help='number of test mixtures, defaults to 10')
    parser.add_argument('--test-rooms', nargs='+',
                        type=bmm.arg_set_type,
                        default=[
                            {'surrey_room_a'},
                        ],
                        help='rooms for the grid of test conditions')
    parser.add_argument('--test-snrs', nargs='+',
                        type=lambda x: bmm.arg_list_type(x, int),
                        default=[
                            [0, 0],
                        ],
                        help='snr limits for the grid of test conditions')
    parser.add_argument('--test-noises', nargs='+',
                        type=bmm.arg_set_type,
                        default=[
                            {'icra_01'},
                            {'icra_02'},
                            {'icra_03'},
                            {'icra_04'},
                            {'icra_05'},
                            {'icra_06'},
                            {'icra_07'},
                            {'icra_08'},
                            {'icra_09'},
                        ],
                        help='noises for the grid of test conditions')
    parser.add_argument('--test-angles', nargs='+',
                        type=lambda x: bmm.arg_list_type(x, int),
                        default=[
                            [0.0, 0.0],
                        ],
                        help='angle limits for the grid of test conditions')
    parser.add_argument('--test-rms',  nargs='+',
                        type=lambda x: bool(int(x)),
                        default=[
                            False,
                        ],
                        help='random rms for the grid of test conditions')
    parser.add_argument('--test-speakers',
                        type=bmm.arg_set_type,
                        default=[
                            {'ieee'},
                        ],
                        help='speakers for the grid of test conditions')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
