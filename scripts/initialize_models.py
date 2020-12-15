import os
import shutil
import itertools

import yaml

from brever.config import defaults
from brever.modelmanagement import (get_unique_id, set_config_field, flatten,
                                    unflatten, arg_to_keys_map,
                                    ModelFilterArgParser, get_config_field)


def check_if_train_and_val_path_exist(configs):
    defaults_ = defaults().to_dict()
    def_train_path = get_config_field(defaults_, 'train_path')
    def_val_path = get_config_field(defaults_, 'val_path')
    for config in configs:
        train_path = get_config_field(config, 'train_path')
        val_path = get_config_field(config, 'val_path')
        if train_path is None and not os.path.exists(def_train_path):
            print('No train path specified, and default path does not exist')
            resp = input(f'Do you wish to continue? y/n')
        elif not os.path.exists(train_path):
            print('The specified train path does not exist')
            resp = input(f'Do you wish to continue? y/n')
        elif val_path is None and not os.path.exists(def_val_path):
            print('No val path specified, and default path does not exist')
            resp = input(f'Do you wish to continue? y/n')
        elif not os.path.exists(val_path):
            print('The specified val path does not exist')
            resp = input(f'Do you wish to continue? y/n')
        else:
            continue
        if resp == 'y':
            return True
        else:
            return False
    return True


def main(args):
    to_combine = {}
    for key in arg_to_keys_map.keys():
        value = args.__getattribute__(key)
        if value is not None:
            set_config_field(to_combine, key, value)

    to_combine = flatten(to_combine)
    keys, values = zip(*to_combine.items())
    configs = unflatten(keys, itertools.product(*values))

    result = check_if_train_and_val_path_exist(configs)
    if not result:
        print('Aborting')
        return

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
                dirpath = os.path.join('models', unique_id)
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
                os.makedirs(dirpath)
                with open(os.path.join(dirpath, 'config.yaml'), 'w') as f:
                    yaml.dump(config, f)
                print(f'Initialized {unique_id}')
        else:
            print('No model was initialized')


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='initialize models')
    args, _ = parser.parse_args()
    main(args)
