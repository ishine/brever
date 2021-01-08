import os

import yaml

from brever.modelmanagement import set_dict_field, DatasetInitArgParser


def main(alias, params, force, n_test):
    if n_test is None:
        n_test = 25

    snrs = [0, 3, 6, 9, 12, 15]
    room_aliases = [
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
    ]

    for snr in snrs:
        for room_alias in room_aliases:
            config = {
                'PRE': {
                    'MIXTURES': {
                        'NUMBER': n_test,
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
                            }
                        }
                    }
                }
            }

            for param, value in params.items():
                if value is not None:
                    key_list = DatasetInitArgParser.arg_to_keys_map[param]
                    set_dict_field(config, key_list, value)

            set_dict_field(
                config,
                ['PRE', 'MIXTURES', 'RANDOM', 'ROOMS'],
                {room_alias},
            )
            set_dict_field(
                config,
                ['PRE', 'MIXTURES', 'RANDOM', 'TARGET', 'SNR', 'MIN'],
                snr,
            )
            set_dict_field(
                config,
                ['PRE', 'MIXTURES', 'RANDOM', 'TARGET', 'SNR', 'MAX'],
                snr,
            )

            letter = room_alias[-1].upper()
            if alias == '':
                dirname = f'testing_snr{snr}_room{letter}'
            else:
                dirname = f'testing_{alias}_snr{snr}_room{letter}'
            dirpath = os.path.join('data', 'processed', dirname)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            config_filepath = os.path.join(dirpath, 'config.yaml')
            if os.path.exists(config_filepath) and not force:
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
    parser.add_argument('--n-test', type=int,
                        help='number of testing mixtures, defaults to 25')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args.alias, params, args.force, args.n_test)
