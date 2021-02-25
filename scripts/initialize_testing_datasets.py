import os

import yaml

from brever.modelmanagement import set_dict_field, DatasetInitArgParser


def main(args, params):

    configs = []
    dirpaths = []

    for snr in args.test_snrs:
        for room_alias in args.test_rooms:
            config = {
                'PRE': {
                    'MIXTURES': {
                        'NUMBER': args.n_test,
                        'DIFFUSE': {
                            'ON': False,
                        },
                        'DECAY': {
                            'ON': False,
                        },
                        'FILELIMITS': {
                            'NOISE': [0.85, 1.0],
                            'TARGET': [0.85, 1.0]
                        },
                        'SAVE': True,
                        'RANDOM': {
                            'SOURCES': {
                                'NUMBER': {
                                    'MIN': 1
                                }
                            },
                            'ROOMS': {room_alias},
                            'TARGET': {
                                'SNR': {
                                    'DISTARGS': [snr, snr],
                                    'DISTNAME': 'uniform'
                                }
                            }
                        }
                    }
                }
            }

            for param, value in params.items():
                if value is not None:
                    key_list = DatasetInitArgParser.arg_to_keys_map[param]
                    set_dict_field(config, key_list, value)

            if args.alias == '':
                dirname = f'testing_snr{snr}_{room_alias}'
            else:
                dirname = f'testing_{args.alias}_snr{snr}_{room_alias}'
            dirpath = os.path.join('data', 'processed', dirname)

            configs.append(config)
            dirpaths.append(dirpath)

    print('The following datasets will be initialized:')
    for dirpath in dirpaths:
        print(dirpath)
    resp = input('Do you with to continue? y/n')
    if resp != 'y':
        return

    for config, dirpath in zip(configs, dirpaths):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        config_filepath = os.path.join(dirpath, 'config.yaml')
        if os.path.exists(config_filepath) and not args.force:
            print(f'{config_filepath} already exists')
            continue
        with open(config_filepath, 'w') as f:
            yaml.dump(config, f)
        print(f'Created {config_filepath}')


if __name__ == '__main__':
    parser = DatasetInitArgParser(description='initialize testing datasets')
    parser.add_argument('alias',
                        help='dataset alias')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('--n-test', type=int, default=10,
                        help='number of testing mixtures, defaults to 25')
    parser.add_argument('--test-rooms', type=str, nargs='+',
                        default=[
                                'surrey_room_a',
                                'surrey_room_b',
                                'surrey_room_c',
                                'surrey_room_d',
                            ],
                        help='rooms for the grid of test condtitions')
    parser.add_argument('--test-snrs', type=int, nargs='+',
                        default=[0, 3, 6, 9, 12, 15],
                        help='snr values for the grid of test conditions')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
