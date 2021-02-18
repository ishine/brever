import os
import shutil
import itertools
import re

import yaml

from brever.config import defaults
from brever.modelmanagement import (get_unique_id, set_config_field, flatten,
                                    unflatten, ModelFilterArgParser,
                                    get_config_field)


def check_if_path_exists(configs, path_type='train'):
    defaults_ = defaults().to_dict()
    default_path = get_config_field(defaults_, f'{path_type}_path')
    for config in configs:
        path = get_config_field(config, f'{path_type}_path')
        if path is None:
            path = default_path
            msg = f'No {path_type} path specified, and default path does not '\
                  'exist'
        else:
            msg = f'The specified {path_type} path does not exist'
        if not os.path.exists(path):
            print(msg)
            resp = input('Do you wish to continue? y/n')
            if resp == 'y':
                return True
            else:
                return False
    return True


def check_trailing_slashes(configs, path_type='train'):
    for config in configs:
        path = get_config_field(config, f'{path_type}_path')
        if path is not None and not path.endswith(('\\', '/')):
            print(f'The specified {path_type} path has no trailing slashes')
            resp = input('Do you wish to continue? y/n')
            if resp == 'y':
                return True
            else:
                return False
    return True


def check_if_test_datasets_exist(configs):
    defaults_ = defaults().to_dict()
    default_path = get_config_field(defaults_, 'test_path')
    for config in configs:
        path = get_config_field(config, 'test_path')
        if path is None:
            path = default_path
            msg = 'No base test path specified, and the default base test '\
                  'path does not correspond to any test dataset directory in '\
                  'the filesystem'
        else:
            msg = 'The specified base test path does not correspond to any '\
                  'existing test dataset directory in the filesystem'
        basename = os.path.basename(path)
        dirname = os.path.dirname(path)
        r = re.compile(fr'^{basename}_(snr-?\d{{1,2}})_(.*)$')
        dirs_ = [dir_ for dir_ in filter(r.match, os.listdir(dirname))]
        if not dirs_:
            print(msg)
            resp = input('Do you wish to continue? y/n')
            if resp == 'y':
                return True
            else:
                return False
    return True


def check_paths(configs):
    return (
        check_if_path_exists(configs, 'train')
        and check_if_path_exists(configs, 'val')
        and check_trailing_slashes(configs, 'train')
        and check_trailing_slashes(configs, 'val')
        and check_if_test_datasets_exist(configs)
    )


def main(args):
    to_combine = {}
    for key in ModelFilterArgParser.arg_to_keys_map.keys():
        value = args.__getattribute__(key)
        if value is not None:
            set_config_field(to_combine, key, value)

    if to_combine:
        to_combine = flatten(to_combine)
        keys, values = zip(*to_combine.items())
        configs = unflatten(keys, itertools.product(*values))
    else:
        configs = [{}]

    result = check_paths(configs)
    if not result:
        print('Aborting')
        return

    new_configs = []
    skipped = 0
    for config in configs:
        unique_id = get_unique_id(config)
        if unique_id not in os.listdir('models'):
            defaults().update(config)  # throws an error if config is not valid
            uni_features = get_config_field(config, 'uni_norm_features', None)
            features = get_config_field(config, 'features', None)
            if (uni_features is not None and features is not None
                    and not uni_features.issubset(features)):
                skipped += 1
            else:
                new_configs.append(config)

    print(f'{len(configs)-skipped} config(s) attempted to be initialized.')
    print(f'{len(configs)-len(new_configs)-skipped} already exist.')

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
