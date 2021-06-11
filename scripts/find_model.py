import os
import shutil

import yaml

from brever.config import defaults
from brever.modelmanagement import (find_model, ModelFilterArgParser,
                                    set_config_field)


def main(delete=False, set_field=None, **kwargs):
    if delete and set_field:
        raise ValueError('Can\'t use both --delete and --set-field')

    models_dir = defaults().PATH.MODELS

    models = find_model(**kwargs)

    tested = []
    trained = []
    untrained = []

    for model_id in models:
        score_file = os.path.join(models_dir, model_id, 'scores.mat')
        loss_file = os.path.join(models_dir, model_id, 'losses.npz')
        if os.path.exists(score_file):
            tested.append(model_id)
        elif os.path.exists(loss_file):
            trained.append(model_id)
        else:
            untrained.append(model_id)

    print(f'{len(models)} total models found')
    print(f'{len(tested)} tested models:')
    for model_id in tested:
        print(model_id)
    print(f'{len(trained)} trained models:')
    for model_id in trained:
        print(model_id)
    print(f'{len(untrained)} untrained models:')
    for model_id in untrained:
        print(model_id)

    if models and delete:
        print(f'{len(models)} models will be deleted.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for model_id in models:
                model_dir = os.path.join(models_dir, model_id)
                shutil.rmtree(model_dir)
                print(f'Deleted {model_dir}')
        else:
            print('No model was deleted')

    if models and set_field:
        tag, value = set_field
        tag = tag.replace('_', '-')
        parser = ModelFilterArgParser()
        args, _ = parser.parse_args([f'--{tag}', value])
        tag = tag.replace('-', '_')
        value, = getattr(args, tag)
        print((f'{len(models)} models will have their {tag} field set to '
               f'{value}.'))
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for model_id in models:
                if model_id in trained or model_id in tested:
                    filenames = ['config.yaml', 'config_full.yaml']
                else:
                    filenames = ['config.yaml']
                for filename in filenames:
                    config_path = os.path.join(models_dir, model_id, filename)
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    set_config_field(config, tag, value)
                    defaults().update(config)  # throw error if conf not valid
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f)
        else:
            print('No model was updated')


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='find models')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='delete found models')
    parser.add_argument('--set-field', nargs=2,
                        help='set a field in config files (use carefully)')
    filter_args, extra_args = parser.parse_args()
    main(delete=extra_args.delete, set_field=extra_args.set_field,
         **vars(filter_args))
