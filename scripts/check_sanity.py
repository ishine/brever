import os
import shutil

from brever.config import defaults
import brever.modelmanagement as bmm


def format_slashes(x):
    x = x.replace('\\', '/')
    x = x.strip('/')
    return x


def shorten(long_str):
    return long_str[:6] + '...' + long_str[-6:]


def main():
    full_configs = []
    full_configs_ids = []
    models_dir = defaults().PATH.MODELS
    sane = True

    for model_id in os.listdir(models_dir):

        model_dir = os.path.join(models_dir, model_id)
        config_filepath = os.path.join(model_dir, 'config.yaml')
        config = bmm.read_yaml(config_filepath)

        train_path = bmm.get_config_field(config, 'train_path')
        val_path = bmm.get_config_field(config, 'val_path')

        if train_path is not None and val_path is not None:
            train_path = format_slashes(train_path)
            val_path = format_slashes(val_path)
            train_basename = os.path.basename(train_path)
            val_basename = os.path.basename(val_path)
            if not train_basename.startswith('train'):
                print(f'Model {shorten(model_id)}\'s train basename does not '
                      f'start with "train"! Got {train_path}')
                sane = False
            if not val_basename.startswith('val'):
                print(f'Model {shorten(model_id)}\'s val basename does not '
                      f'start with "val"! Got {val_path}')
                sane = False
            train_alias = train_path.replace(train_basename, '')
            val_alias = val_path.replace(val_basename, '')
            if train_alias != val_alias:
                print(f'Model {shorten(model_id)}\'s train and val paths are '
                      f'unconsistent! Got basenames {train_basename} and '
                      f'{val_basename}')
                sane = False
        elif train_path is not None:
            print(f'Model {shorten(model_id)} has a train path but no val '
                  'path!')
            sane = False
        elif val_path is not None:
            print(f'Model {shorten(model_id)} has a val path but no train '
                  'path!')
            sane = False

        full_config = defaults()
        full_config.update(config)
        full_config = full_config.to_dict()
        if full_config in full_configs:
            index = full_configs.index(full_config)
            duple_id = full_configs_ids[index]
            duple_dir = os.path.join(models_dir, duple_id)
            print(f'Models {shorten(model_id)} and {shorten(duple_id)} '
                  f'are duplicates!')
            sane = False
            scores = os.path.join(model_dir, 'scores.mat')
            scores_dup = os.path.join(duple_dir, 'scores.mat')
            if not os.path.exists(scores) or os.path.exists(scores_dup):
                if not os.path.exists(scores):
                    print(f'Model {shorten(model_id)} is untrained')
                elif os.path.exists(scores_dup):
                    print(f'Models {shorten(model_id)} and '
                          f'{shorten(duple_id)} are both trained')
                resp = input(f'Would you like to delete model '
                             f'{shorten(model_id)}? y/n')
                if resp == 'y':
                    shutil.rmtree(model_dir)
                    print(f'Deleted {shorten(model_id)}')
                    continue
                else:
                    print('Model was not deleted')
            else:
                print(f'Model {shorten(duple_id)} is untrained')
                resp = input(f'Would you like to delete model '
                             f'{shorten(duple_id)}? y/n')
                if resp == 'y':
                    shutil.rmtree(duple_dir)
                    print(f'Deleted model {shorten(duple_id)}')
                    full_configs[index] = full_config
                    full_configs_ids[index] = model_id
                else:
                    print('Model was not deleted')
        else:
            full_configs.append(full_config)
            full_configs_ids.append(model_id)

        new_id = bmm.get_unique_id(config)
        if new_id != model_id:
            print(f'Model {shorten(model_id)} has wrong ID!')
            sane = False
            resp = input('Would you like to rename it? y/n')
            if resp == 'y':
                new_dir = os.path.join(models_dir, new_id)
                os.rename(model_dir, new_dir)
                print(f'Renamed model {shorten(model_id)} to {new_id}')
            else:
                print('Model was not renamed')

    if sane:
        print('Models directory is sane')


if __name__ == '__main__':
    main()
