import os
import shutil
import itertools

import yaml

from brever.config import defaults
from brever.modelmanagement import (get_unique_id, set_dict_field, flatten,
                                    unflatten, arg_to_keys_map,
                                    ModelFilterArgParser)


def main(args):
    to_combine = {}
    for attr, key_list in arg_to_keys_map.items():
        value = args.__getattribute__(attr)
        if value is not None:
            set_dict_field(to_combine, key_list, value)

    to_combine = flatten(to_combine)
    keys, values = zip(*to_combine.items())
    configs = unflatten(keys, itertools.product(*values))

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
