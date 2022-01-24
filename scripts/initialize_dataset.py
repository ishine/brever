import os

from brever.config import defaults
import brever.management as bm


def main(args, params):
    processed_dir = defaults().PATH.PROCESSED

    if sum((args.train, args.val, args.test)) == 0:
        raise ValueError('must provide one of --train, --val and --test')
    elif sum((args.train, args.val, args.test)) > 1:
        raise ValueError('can only provide one of --train, --val and --test')

    if args.train:
        processed_dir = os.path.join(processed_dir, 'train')
    elif args.val:
        processed_dir = os.path.join(processed_dir, 'val')
    elif args.test:
        processed_dir = os.path.join(processed_dir, 'test')

    config = {}
    for param, value in params.items():
        if value is not None:
            key_list = bm.DatasetInitArgParser.arg_to_keys_map[param]
            bm.set_dict_field(config, key_list, value)

    if args.name is None:
        args.name = bm.get_unique_id(config)

    dirpath = os.path.join(processed_dir, args.name)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    config_filepath = os.path.join(dirpath, 'config.yaml')
    if os.path.exists(config_filepath) and not args.force:
        print(f'Already exists! {config_filepath} ')
    else:
        bm.dump_yaml(config, config_filepath)
        print(f'Created {config_filepath}')


if __name__ == '__main__':
    parser = bm.DatasetInitArgParser(description='initialize dataset')
    parser.add_argument('--name', type=str,
                        help='dataset name')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('--train', action='store_true',
                        help='dump in train subdir')
    parser.add_argument('--val', action='store_true',
                        help='dump in val subdir')
    parser.add_argument('--test', action='store_true',
                        help='dump in test subdir')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
