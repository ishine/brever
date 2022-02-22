import argparse
import os

from brever.config import get_config


def main():
    models_dir = get_config('config/paths.yaml').MODELS
    yes_to_all = False

    for model_id in os.listdir(models_dir):

        model_dir = os.path.join(models_dir, model_id)
        config_path = os.path.join(model_dir, 'config.yaml')
        config = get_config(config_path)
        new_id = config.get_hash()

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
