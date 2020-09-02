import yaml
import os

from brever.modelmanagement import get_unique_id


for model_id in os.listdir('models'):
    model_dirpath = os.path.join('models', model_id)
    config_filepath = os.path.join(model_dirpath, 'config.yaml')
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
    train_path = config['POST']['PATH']['TRAIN']
    val_path = config['POST']['PATH']['VAL']
    test_path = config['POST']['PATH']['TEST']
    if (train_path == 'data\\processed\\training'
            and val_path == 'data\\processed\\validation'
            and test_path == 'data\\processed\\testing'):
        pass
    elif (train_path == 'data\\processed\\centered_training'
            and val_path == 'data\\processed\\centered_validation'
            and test_path == 'data\\processed\\centered_testing'):
        pass
    elif (train_path == 'data\\processed\\onlyreverb_training'
            and val_path == 'data\\processed\\onlyreverb_validation'
            and test_path == 'data\\processed\\onlyreverb_testing'):
        pass
    elif (train_path == 'data\\processed\\training'
            and val_path == 'data\\processed\\validation'
            and test_path == 'data\\processed\\testing_big'):
        pass
    else:
        print(f'Model {model_id} has unconsistent dataset paths:')
        print(train_path)
        print(val_path)
        print(test_path)
    new_id = get_unique_id(config)
    if new_id != model_id:
        print(f'Model {model_id} has wrong ID!!!')
        resp = input('Would you like to rename it? y/n')
        if resp == 'y':
            new_dirpath = os.path.join('models', new_id)
            os.rename(model_dirpath, new_dirpath)
            print(f'Renamed {model_id} to {new_id}')
        else:
            print('Model was not renamed')
