import os
import yaml

from brever.args import DatasetArgParser
from brever.config import get_config


def main():
    paths = get_config('config/paths.yaml')

    config = get_config('config/dataset.yaml')
    config.update_from_args(args, parser.arg_map)
    dataset_id = config.get_hash()

    dataset_dir = os.path.join(paths.DATASETS, args.kind, dataset_id)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    config_path = os.path.join(dataset_dir, 'config.yaml')
    if os.path.exists(config_path) and not args.force:
        raise FileExistsError(f'dataset already exists: {config_path} ')
    else:
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f)
        print(f'Initialized {config_path}')


if __name__ == '__main__':
    parser = DatasetArgParser(description='initialize a dataset')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('kind', choices=['train', 'test'],
                        help='dump in train or test subdir')
    args = parser.parse_args()
    main()
