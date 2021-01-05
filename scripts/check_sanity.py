import os
import shutil

import yaml

from brever.config import defaults
from brever.modelmanagement import get_unique_id, get_config_field


def format_slashes(x):
    x = x.replace('\\', '/')
    x = x.strip('/')
    return x


def shorten(long_str):
    return long_str[:6] + '...'


def main():
    full_configs = []
    full_configs_ids = []

    for model_id in os.listdir('models'):
        model_dirpath = os.path.join('models', model_id)
        config_filepath = os.path.join(model_dirpath, 'config.yaml')
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)

        train_path = get_config_field(config, 'train_path')
        val_path = get_config_field(config, 'val_path')

        if train_path is not None and val_path is not None:
            train_path = format_slashes(train_path)
            val_path = format_slashes(val_path)
            base_train_path = os.path.join('data', 'processed', 'training')
            base_val_path = os.path.join('data', 'processed', 'validation')
            base_train_path = format_slashes(base_train_path)
            base_val_path = format_slashes(base_val_path)
            if not train_path.startswith(base_train_path):
                print((f'Model {shorten(model_id)} has wrong train path! '
                       f'{train_path}'))
            if not val_path.startswith(base_val_path):
                print((f'Model {shorten(model_id)} has wrong val path! '
                       f'{val_path}'))
            train_path_strip = train_path.replace(base_train_path, '')
            val_path_strip = val_path.replace(base_val_path, '')
            if train_path_strip != val_path_strip:
                print((f'Model {shorten(model_id)} has unconsistent datasets '
                       f'paths!'))
                print(f'{train_path_strip}')
                print(f'{val_path_strip}')
        elif train_path is not None or val_path is not None:
            print((f'Model {shorten(model_id)} has unconsistent datasets '
                   f'paths!'))
            print(f'{train_path}')
            print(f'{val_path}')

        full_config = defaults()
        full_config.update(config)
        full_config = full_config.to_dict()
        if full_config in full_configs:
            index = full_configs.index(full_config)
            duplicate_id = full_configs_ids[index]
            print((f'Models {shorten(model_id)} and {shorten(duplicate_id)} '
                   f'are duplicates'))
            pesqfile = os.path.join('models', model_id, 'pesq_scores.mat')
            pesqfile_ = os.path.join('models', duplicate_id, 'pesq_scores.mat')
            if not os.path.exists(pesqfile) or os.path.exists(pesqfile_):
                if not os.path.exists(pesqfile):
                    print((f'Model {shorten(model_id)} is untrained'))
                elif os.path.exists(pesqfile_):
                    print((f'Models {shorten(model_id)} and '
                           f'{shorten(duplicate_id)} are both trained'))
                resp = input((f'Would you like to delete model '
                              f'{shorten(model_id)}? y/n'))
                if resp == 'y':
                    shutil.rmtree(model_dirpath)
                    print(f'Deleted {shorten(model_id)}')
                    continue
                else:
                    print('Model was not deleted')
            else:
                print((f'Model {shorten(duplicate_id)} is untrained'))
                resp = input((f'Would you like to delete model '
                              f'{shorten(duplicate_id)}? y/n'))
                if resp == 'y':
                    model_dirpath_ = os.path.join('models', duplicate_id)
                    shutil.rmtree(model_dirpath_)
                    print(f'Deleted model {shorten(duplicate_id)}')
                    full_configs[index] = full_config
                    full_configs_ids[index] = model_id
                else:
                    print('Model was not deleted')
        else:
            full_configs.append(full_config)
            full_configs_ids.append(model_id)

        new_id = get_unique_id(config)
        if new_id != model_id:
            print(f'Model {shorten(model_id)} has wrong ID!')
            resp = input('Would you like to rename it? y/n')
            if resp == 'y':
                new_dirpath = os.path.join('models', new_id)
                os.rename(model_dirpath, new_dirpath)
                print(f'Renamed model {shorten(model_id)} to {new_id}')
            else:
                print('Model was not renamed')


if __name__ == '__main__':
    main()
