import yaml
import os

from brever.modelmanagement import get_unique_id, get_dict_field


def format_slashes(x):
    x = x.replace('\\', '/')
    x = x.strip('/')
    return x


def main():
    for model_id in os.listdir('models'):
        model_dirpath = os.path.join('models', model_id)
        config_filepath = os.path.join(model_dirpath, 'config.yaml')
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)

        train_path = get_dict_field(config, ['POST', 'PATH', 'TRAIN'])
        val_path = get_dict_field(config, ['POST', 'PATH', 'VAL'])

        if train_path is not None and val_path is not None:
            train_path = format_slashes(train_path)
            val_path = format_slashes(val_path)
            base_train_path = os.path.join('data', 'processed', 'training')
            base_val_path = os.path.join('data', 'processed', 'validation')
            base_train_path = format_slashes(base_train_path)
            base_val_path = format_slashes(base_val_path)
            if not train_path.startswith(base_train_path):
                print(f'Model {model_id} has wrong train path! {train_path}')
            if not val_path.startswith(base_val_path):
                print(f'Model {model_id} has wrong val path! {val_path}')
            train_path_strip = train_path.replace(base_train_path, '')
            val_path_strip = val_path.replace(base_val_path, '')
            if train_path_strip != val_path_strip:
                print(f'Model {model_id} has unconsistent datasets paths!')
                print(f'{train_path_strip}')
                print(f'{val_path_strip}')
        elif train_path is not None or val_path is not None:
            print(f'Model {model_id} has unconsistent datasets paths!')
            print(f'{train_path}')
            print(f'{val_path}')

        new_id = get_unique_id(config)
        if new_id != model_id:
            print(f'Model {model_id} has wrong ID!')
            resp = input('Would you like to rename it? y/n')
            if resp == 'y':
                new_dirpath = os.path.join('models', new_id)
                os.rename(model_dirpath, new_dirpath)
                print(f'Renamed {model_id} to {new_id}')
            else:
                print('Model was not renamed')


if __name__ == '__main__':
    main()
