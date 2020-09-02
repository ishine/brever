import os
import shutil
import copy

import yaml

from brever.config import defaults
from brever.modelmanagement import get_unique_id, set_dict_field


def main():
    base_config = {
        'MODEL': {
            'BATCHNORM': {
                'ON': False,
                'MOMENTUM': 0.1,
            },
            'DROPOUT': {
                'ON': True,
                'RATE': 0.2,
            },
            'NLAYERS': 1,
            'WEIGHTDECAY': 0,
            'LEARNINGRATE': 1e-4,
            'BATCHSIZE': 32,
        },
        'POST': {
            'STACK': 4,
            'FEATURES': ['pdf'],
            'PATH': {
                'TRAIN': 'data\\processed\\training',
                'VAL': 'data\\processed\\validation',
                'TEST': 'data\\processed\\testing_big',
            },
            'DECIMATION': 2,
        }
    }

    configs = [copy.deepcopy(base_config)]
    keys_values = [
        # (['MODEL', 'BATCHNORM', 'ON'], [False, True]),
        # (['MODEL', 'DROPOUT', 'ON'], [False, True]),
        # (['MODEL', 'NLAYERS'], [1, 2, 3]),
        # (['POST', 'STACK'], [0, 1, 2, 3, 4]),
        # (['POST', 'FEATURES'], [
        #     ['ild'],
        #     ['itd'],
        #     ['ic'],
        #     ['mfcc'],
        #     ['pdf'],
        #     ['logpdf'],
        #     ['ild', 'itd', 'ic'],
        #     ['mfcc', 'pdf'],
        #     ['mfcc', 'logpdf'],
        #     ['ic', 'mfcc', 'pdf'],
        #     ['ic', 'mfcc', 'logpdf'],
        # ]),
        # (['MODEL', 'BATCHSIZE'], [32, 1024]),
        # (['POST', 'PATH'], [
        #         {
        #             'TRAIN': 'data\\processed\\training',
        #             'VAL': 'data\\processed\\validation',
        #             'TEST': 'data\\processed\\testing',
        #         },
        #         {
        #             'TRAIN': 'data\\processed\\centered_training',
        #             'VAL': 'data\\processed\\centered_validation',
        #             'TEST': 'data\\processed\\centered_testing',
        #         },
        #         {
        #             'TRAIN': 'data\\processed\\onlyreverb_training',
        #             'VAL': 'data\\processed\\onlyreverb_validation',
        #             'TEST': 'data\\processed\\onlyreverb_testing',
        #         },
        #     ]),
    ]
    for keys, values in keys_values:
        for val in values:
            config = copy.deepcopy(base_config)
            set_dict_field(config, keys, val)
            if config not in configs:
                configs.append(config)

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
                dirpath = f'models\\{unique_id}'
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
                os.makedirs(dirpath)
                with open(os.path.join(dirpath, 'config.yaml'), 'w') as f:
                    yaml.dump(config, f)
                print(f'Initialized {unique_id}')
        else:
            print('No model was initialized')


if __name__ == '__main__':
    main()
