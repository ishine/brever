import shutil
import os

import yaml

from brever.modelmanagement import ModelFilterArgParser, set_config_field, get_unique_id


def shorten(long_str):
    return long_str[:6] + '...'


def main(args, params):
    model = args.input

    for key, values in params.items():
        if values is not None:
            for val in values:
                config_file = os.path.join(model, 'config.yaml')
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                set_config_field(config, key, val)

                new_id = get_unique_id(config)

                dst = os.path.join('models', new_id)

                if args.force and os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(model, dst)
                print(f'Copied to {dst}')

                new_config_file = os.path.join(dst, 'config.yaml')
                with open(new_config_file, 'w') as f:
                    yaml.dump(config, f)

                config_file = os.path.join(model, 'config_full.yaml')
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                set_config_field(config, key, val)
                new_config_file = os.path.join(dst, 'config_full.yaml')
                with open(new_config_file, 'w') as f:
                    yaml.dump(config, f)

                os.remove(os.path.join(dst, 'scores.npz'))
                os.remove(os.path.join(dst, 'scores.mat'))


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='plot feature distribution')
    parser.add_argument('input', help='input model')
    parser.add_argument('--force', action='store_true')
    model_args, args = parser.parse_args()
    params = vars(model_args)
    main(args, params)
