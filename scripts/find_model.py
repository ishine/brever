import os
import shutil

import yaml

from brever.config import defaults
import brever.modelmanagement as bmm


def main(delete=False, set_field=None, **kwargs):
    if delete and set_field:
        raise ValueError('Can\'t use both --delete and --set-field')

    models = bmm.find_model(**kwargs)

    tested = []
    trained = []
    untrained = []

    for model in models:
        score_file = os.path.join(model, 'scores.mat')
        loss_file = os.path.join(model, 'losses.npz')
        if os.path.exists(score_file):
            tested.append(model)
        elif os.path.exists(loss_file):
            trained.append(model)
        else:
            untrained.append(model)

    print(f'{len(models)} total models found')
    print(f'{len(tested)} tested models:')
    for model in tested:
        print(model)
    print(f'{len(trained)} trained models:')
    for model in trained:
        print(model)
    print(f'{len(untrained)} untrained models:')
    for model in untrained:
        print(model)

    if models and delete:
        print(f'{len(models)} models will be deleted.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for model in models:
                shutil.rmtree(model)
                print(f'Deleted {model}')
        else:
            print('No model was deleted')

    if models and set_field:
        tag, value = set_field
        tag = tag.replace('_', '-')
        parser = bmm.ModelFilterArgParser()
        args, _ = parser.parse_args([f'--{tag}', value])
        tag = tag.replace('-', '_')
        value, = getattr(args, tag)
        print(f'{len(models)} models will have their {tag} field set to '
              f'{value}.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            def_cfg = defaults()
            for model in models:
                if model in trained or model in tested:
                    filenames = ['config.yaml', 'config_full.yaml']
                else:
                    filenames = ['config.yaml']
                for filename in filenames:
                    config_path = os.path.join(model, filename)
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    bmm.set_config_field(config, tag, value)
                    def_cfg.update(config)  # throw error if conf not valid
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f)
        else:
            print('No model was updated')


if __name__ == '__main__':
    parser = bmm.ModelFilterArgParser(description='find models')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='delete found models')
    parser.add_argument('--set-field', nargs=2,
                        help='set a field in config files (use carefully)')
    filter_args, extra_args = parser.parse_args()
    main(delete=extra_args.delete, set_field=extra_args.set_field,
         **vars(filter_args))
