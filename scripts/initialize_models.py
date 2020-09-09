import os
import shutil
import argparse
import itertools

import yaml

from brever.config import defaults
from brever.modelmanagement import get_unique_id, set_dict_field, flatten, unflatten


def main(args):
    to_combine = {}
    for attr, key_list in [
                ('layers', ['MODEL', 'NLAYERS']),
                ('stacks', ['POST', 'STACK']),
                ('batchnorm', ['MODEL', 'BATCHNORM', 'ON']),
                ('dropout', ['MODEL', 'DROPOUT', 'ON']),
                ('batchsize', ['MODEL', 'BATCHSIZE']),
                ('features', ['POST', 'FEATURES']),
                ('train_path', ['POST', 'PATH', 'TRAIN']),
                ('val_path', ['POST', 'PATH', 'VAL']),
            ]:
        value = args.__getattribute__(attr)
        if value is not None:
            value = list(dict.fromkeys(value))
            if attr == 'features':
                value = [set(item.split(' ')) for item in value]
            set_dict_field(to_combine, key_list, value)

    to_combine = flatten(to_combine)
    keys, values = zip(*to_combine.items())
    configs = unflatten(keys, itertools.product(*values))

    [print(config) for config in configs]

    new_configs = []
    for config in configs:
        unique_id = get_unique_id(config)
        if unique_id not in os.listdir('models'):
            defaults().update(config)  # throws an error if config is not valid
            new_configs.append(config)

    print(f'{len(configs)} config(s) attempted to be initialized.')
    print(f'{len(configs)-len(new_configs)} already exist.')

    if not new_configs:
        print(f'{len(new_configs)} will be initialized.')
    else:
        resp = input(f'{len(new_configs)} will be initialized. Continue? y/n')
        if resp == 'y':
            for config in new_configs:
                unique_id = get_unique_id(config)
                dirpath = f'models\\{unique_id}'
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
                os.makedirs(dirpath)
                with open(os.path.join(dirpath, 'config.yaml'), 'w') as f:
                    yaml.dump(config, f)
                print(f'Initialized {unique_id}')
        else:
            print('No model was initialized')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find models.')
    parser.add_argument('--layers', type=int, nargs='+',
                        help=('number of layers'))
    parser.add_argument('--stacks', type=int, nargs='+',
                        help=('number of extra stacks'))
    parser.add_argument('--batchnorm', type=lambda x: bool(int(x)), nargs='+',
                        help=('batchnorm toggle'))
    parser.add_argument('--dropout', type=lambda x: bool(int(x)), nargs='+',
                        help=('dropout toggle'))
    parser.add_argument('--batchsize', type=int, nargs='+',
                        help=('mini-batch size'))
    parser.add_argument('--features', nargs='+',
                        help=('feature set'))
    parser.add_argument('--train-path', nargs='+',
                        help=('training dataset path'))
    parser.add_argument('--val-path', nargs='+',
                        help=('validation dataset path'))
    args = parser.parse_args()
    main(args)
