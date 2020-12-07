import os
import copy
import argparse

import yaml

from brever.modelmanagement import set_dict_field


def main(force, dirpath_target):
    base_config = {
        'PRE': {
            'MIXTURES': {
                'NUMBER': 25,
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

            if dirpath_target is not None:
                set_dict_field(
                    config,
                    ['PRE', 'MIXTURES', 'PATH', 'TARGET'],
                    dirpath_target,
                )

            letter = room_alias[-1].upper()
            dirname = f'testing_snr{snr}_room{letter}'
            dirpath = os.path.join('data', 'processed', dirname)
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            config_filepath = os.path.join(dirpath, 'config.yaml')
            if os.path.exists(config_filepath) and not force:
                print(f'{config_filepath} already exists')
                continue
            with open(config_filepath, 'w') as f:
                yaml.dump(config, f)
            print(f'Created {config_filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='initialize testing datasets')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('--dirpath-target', '--force', action='store_true',
                        help='path to target speech database')
    args = parser.parse_args()
    main(args.force, args.dirpath_target)
