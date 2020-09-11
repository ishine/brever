import yaml
import os

from brever.modelmanagement import get_unique_id


for model_id in os.listdir('models'):
    model_dirpath = os.path.join('models', model_id)
    config_filepath = os.path.join(model_dirpath, 'config.yaml')
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
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
