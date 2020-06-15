import os

import yaml

from brever.modelmanagement import get_unique_id


models_dir = 'models'
for model_id in os.listdir(models_dir):
    model_dir = os.path.join(models_dir, model_id)
    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file) as f:
        data = yaml.safe_load(f)
    if 'pdf' in data['POST']['FEATURES']:
        data['POST']['FEATURES'].remove('pdf')
        data['POST']['FEATURES'].append('logpdf')
        new_id = get_unique_id(data)
        new_dir = os.path.join(models_dir, new_id)
        new_file = os.path.join(new_dir, 'config.yaml')
        os.makedirs(new_dir)
        with open(new_file, 'w') as f:
            yaml.dump(data, f)
        print(f'Initialized {new_id}')
