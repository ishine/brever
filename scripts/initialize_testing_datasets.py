import os
import copy
import argparse

import yaml

from brever.modelmanagement import set_dict_field


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find models.')
    parser.add_argument('--centered', action='store_true',
                        help=('Centered target.'))
    parser.add_argument('--onlyreverb', action='store_true',
                        help=('Only reverb.'))
    args = parser.parse_args()

    base_config = {
        'PRE': {
            'SEED': {
                'ON': False,
            },
            'MIXTURES': {
                'NUMBER': 10,
                'RANDOM': {
                    'DIFFUSE': {
                        'TYPES': []
                    },
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

    if args.centered:
        basename = 'centered_'
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                     'ANGLE', 'MIN'], 0)
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                     'ANGLE', 'MAX'], 0)

    elif args.onlyreverb:
        basename = 'onlyreverb_'
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                     'ANGLE', 'MIN'], 0)
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                     'ANGLE', 'MAX'], 0)
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES',
                                     'NUMBER', 'MIN'], 0)
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES',
                                     'NUMBER', 'MAX'], 0)
    else:
        basename = ''

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
            set_dict_field(config, ['PRE', 'MIXTURES', 'RANDOM', 'ROOMS'],
                           [room_alias])
            set_dict_field(config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                    'SNR', 'MIN'], snr)
            set_dict_field(config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                    'SNR', 'MAX'], snr)
            letter = room_alias[-1].upper()
            dirname = f'{basename}testing_snr{snr}_room{letter}'
            dirpath = os.path.join('data', 'processed', dirname)
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            config_filepath = os.path.join(dirpath, 'config.yaml')
            with open(config_filepath, 'w') as f:
                yaml.dump(config, f)
