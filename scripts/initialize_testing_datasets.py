import os
import copy
import argparse

import yaml

from brever.modelmanagement import set_dict_field, get_dict_field


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find models.')
    parser.add_argument('--centered', action='store_true',
                        help=('Centered target.'))
    parser.add_argument('--onlyreverb', action='store_true',
                        help=('Only reverb.'))
    parser.add_argument('--onlydiffuse', action='store_true',
                        help=('Only diffuse noise.'))
    parser.add_argument('--big', action='store_true',
                        help=('Set 50 mixtures per condition instead of 10.'))
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

    snrs = [0, 3, 6, 9, 12, 15]
    room_aliases = [
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
    ]
    basename = ''

    if args.big:
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'NUMBER'], 50)

    if args.onlyreverb:
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES',
                                     'NUMBER', 'MIN'], 0)
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES',
                                     'NUMBER', 'MAX'], 0)
        n = get_dict_field(base_config, ['PRE', 'MIXTURES', 'NUMBER'])
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'NUMBER'], 6*n)
        snrs = [0]
        basename = basename + 'onlyreverb_'

    elif args.onlydiffuse:
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES',
                                     'NUMBER', 'MIN'], 0)
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES',
                                     'NUMBER', 'MAX'], 0)
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'DIFFUSE',
                                     'TYPES'], ['noise_white'])
        n = get_dict_field(base_config, ['PRE', 'MIXTURES', 'NUMBER'])
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'NUMBER'], 4*n)
        room_aliases = ['surrey_anechoic']
        basename = basename + 'onlydiffuse_'

    if args.centered:
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                     'ANGLE', 'MIN'], 0)
        set_dict_field(base_config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                     'ANGLE', 'MAX'], 0)
        basename = basename + 'centered_'

    for snr in snrs:
        for room_alias in room_aliases:
            config = copy.deepcopy(base_config)
            set_dict_field(config, ['PRE', 'MIXTURES', 'RANDOM', 'ROOMS'],
                           [room_alias])
            set_dict_field(config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                    'SNR', 'MIN'], snr)
            set_dict_field(config, ['PRE', 'MIXTURES', 'RANDOM', 'TARGET',
                                    'SNR', 'MAX'], snr)
            if not args.onlydiffuse:
                letter = room_alias[-1].upper()
                if args.big:
                    dirname = f'{basename}testing_big_snr{snr}_room{letter}'
                else:
                    dirname = f'{basename}testing_snr{snr}_room{letter}'
            else:
                _, short_alias = room_alias.split('_')
                if args.big:
                    dirname = f'{basename}testing_big_snr{snr}_{short_alias}'
                else:
                    dirname = f'{basename}testing_snr{snr}_{short_alias}'
            dirpath = os.path.join('data', 'processed', dirname)
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            config_filepath = os.path.join(dirpath, 'config.yaml')
            if os.path.exists(config_filepath):
                print(f'File {config_filepath} already exists')
                resp = input('Would you like to overwrite it? y/n')
                if resp != 'y':
                    continue
            with open(config_filepath, 'w') as f:
                yaml.dump(config, f)
            print(f'Initialized {config_filepath}')
