import argparse
import os

from brever.config import defaults
import brever.management as bm


def main():
    models_dir = defaults().PATH.MODELS
    yes_to_all = False

    for model_id in os.listdir(models_dir):

        model_dir = os.path.join(models_dir, model_id)
        config_filepath = os.path.join(model_dir, 'config.yaml')
        config = bm.read_yaml(config_filepath)

        train_path = bm.get_config_field(config, 'train_path')
        val_path = bm.get_config_field(config, 'val_path')

        if train_path is not None and val_path is not None:
            if not os.path.dirname(train_path).endswith('train'):
                print(f'Model {model_id} train path does not point to the '
                      f'train dir! Got {train_path}')
            if not os.path.dirname(val_path).endswith('val'):
                print(f'Model {model_id} val path does not point to the '
                      f'val dir! Got {val_path}')
            if os.path.basename(train_path) != os.path.basename(val_path):
                print(f'Model {model_id} train and val paths are not '
                      f'consistent! Got {train_path} and {val_path}')
            if not os.path.exists(train_path):
                print(f'Model {model_id} train path does not exist!')
            if not os.path.exists(val_path):
                print(f'Model {model_id} val path does not exist!')
        elif train_path is not None:
            print(f'Model {model_id} has a train path but no val path!')
        elif val_path is not None:
            print(f'Model {model_id} has a val path but no train path!')

        new_id = bm.get_unique_id(config)
        if new_id != model_id:
            print(f'Model {model_id} has wrong ID!')
            while True:
                if yes_to_all:
                    r = 'y'
                else:
                    r = input('Would you like to rename it? [y/n/yes-all]')
                if r.lower() in ['y', 'yes-all']:
                    if r.lower() == 'yes-all':
                        yes_to_all = True
                    new_dir = os.path.join(models_dir, new_id)
                    os.rename(model_dir, new_dir)
                    print(f'Renamed model {model_id} to {new_id}')
                    break
                elif r.lower() == 'n':
                    print('Model was not renamed')
                    break
                else:
                    print('Could not interpret answer')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check sanity of model '
                                                 'directory')
    args = parser.parse_args()
    main()
