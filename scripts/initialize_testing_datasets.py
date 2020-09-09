import os
import copy
import argparse

import yaml

from brever.modelmanagement import set_dict_field, get_dict_field


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find models.')
    parser.add_argument('-f', '--force', action='store_true',
                        help=('Overwrite config file if already exists.'))
    args = parser.parse_args()

    base_config = {
        'PRE': {
            'SEED': {
                'ON': False,
            },
            'MIXTURES': {
                'NUMBER': 10,
                'DIFFUSE': {
                    'ON': False,
                },
                'RANDOM': {
                    'SOURCES': {
                        'NUMBER': {
                            'MIN': 1
                        }
                    },
                },
                'FILELIMITS': {
                    'NOISE': [0.85, 1.0],
                    'TARGET': [0.85, 1.0]
                },
                'SAVE': True
            }
        }
    }

    snrs = [0, 3, 6, 9, 12, 15]
    room_aliases = [
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
    ]

    for snr in snrs:
        for room_alias in room_aliases:
            config = copy.deepcopy(base_config)
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
            dirname = f'testing_snr{snr}_room{letter}'
            dirpath = os.path.join('data', 'processed', dirname)
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            config_filepath = os.path.join(dirpath, 'config.yaml')
            if os.path.exists(config_filepath):
                print(f'{config_filepath} already exists')
                if args.force:
                    print('Overwriting')
                else:
                    continue
            with open(config_filepath, 'w') as f:
                yaml.dump(config, f)
            print(f'Initialized {config_filepath}')
