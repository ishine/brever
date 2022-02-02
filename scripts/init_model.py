import os
import yaml

from brever.args import ModelArgParser
from brever.config import get_config, BreverConfig


def main():
    paths = get_config('config/paths.yaml')

    model_config = get_config(f'config/models/{args.arch}.yaml')
    model_config.update_from_args(args, parser.model_arg_map[args.arch])

    train_config = get_config('config/training.yaml')
    train_config.update_from_args(args, parser.training_arg_map)

    config = BreverConfig({
        'MODEL': model_config.to_dict(),
        'TRAINING': train_config.to_dict(),
    })
    model_id = config.get_hash()

    model_dir = os.path.join(paths.MODELS, model_id)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = os.path.join(model_dir, 'config.yaml')
    if os.path.exists(config_path) and not args.force:
        raise FileExistsError(f'model already exists: {config_path} ')
    else:
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f)
        print(f'Initialized {config_path}')


if __name__ == '__main__':
    parser = ModelArgParser(description='initialize a model')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    args = parser.parse_args()
    main()
