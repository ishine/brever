import os
import argparse

import yaml

from brever.modelmanagement import set_dict_field, DatasetInitArgParser


def main(alias, params, force, n_train, n_val):
    if n_train is None:
        n_train = 1000
    if n_val is None:
        n_val = 200
    for basename, filelims, number in [
                ('training', [0.0, 0.7], n_train),
                ('validation', [0.7, 0.85], n_val),
            ]:
        config = {
            'PRE': {
                'MIXTURES': {
                    'NUMBER': number,
                    'FILELIMITS': {
                        'NOISE': filelims.copy(),
                        'TARGET': filelims.copy(),
                    },
                    'RANDOM': {
                        'ROOMS': {'surrey_anechoic'},
                    }
                }
            }
        }

        arg_to_keys_map = {
            'decay': ['PRE', 'MIXTURES', 'DECAY', 'ON'],
            'decay_color': ['PRE', 'MIXTURES', 'DECAY', 'COLOR'],
            'diffuse': ['PRE', 'MIXTURES', 'DIFFUSE', 'ON'],
            'diffuse_color': ['PRE', 'MIXTURES', 'DIFFUSE', 'COLOR'],
            'drr_min': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'DRR', 'MIN'],
            'drr_max': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'DRR', 'MAX'],
            'rt60_min': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'RT60', 'MIN'],
            'rt60_max': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'RT60', 'MAX'],
            'delay_min': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'DELAY', 'MIN'],
            'delay_max': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'DELAY', 'MAX'],
            'rooms': ['PRE', 'MIXTURES', 'RANDOM', 'ROOMS'],
        }
        for param, value in params.items():
            if value is not None:
                key_list = arg_to_keys_map[param]
                set_dict_field(config, key_list, value)

        dirname = f'{basename}_{alias}'
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
    parser = DatasetInitArgParser(description='initialize training datasets')
    parser.add_argument('alias',
                        help='dataset alias')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('--n-train', type=int,
                        help='number of training mixture, defaults to 1000')
    parser.add_argument('--n-val', type=int,
                        help='number of validation mixture, defaults to 200')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args.alias, params, args.force, args.n_train, args.n_val)
