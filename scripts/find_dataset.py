import os
import shutil

from brever.config import defaults
import brever.modelmanagement as bmm


def main(args, **kwargs):
    processed_dir = defaults().PATH.PROCESSED

    if sum((args.train, args.val, args.test)) > 1:
        raise ValueError('can only provide one of --train, --val and --test')

    if args.train:
        processed_dir = os.path.join(processed_dir, 'train')
    elif args.val:
        processed_dir = os.path.join(processed_dir, 'val')
    elif args.test:
        processed_dir = os.path.join(processed_dir, 'test')

    dsets = []
    for root, folder, files in os.walk(processed_dir):
        if 'config.yaml' in files:
            config_file = os.path.join(root, 'config.yaml')
            config = bmm.read_yaml(config_file)
            valid = True
            for key, value in kwargs.items():
                keys = bmm.DatasetInitArgParser.arg_to_keys_map[key]
                if value is not None:
                    if bmm.get_dict_field(config, keys) != value:
                        valid = False
                        break
            if valid:
                dsets.append(root)

    for dset in dsets:
        print(dset)

    if dsets and args.delete:
        print(f'{len(dsets)} datasets will be deleted.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for dset in dsets:
                shutil.rmtree(dset)
                print(f'Deleted {dset}')
        else:
            print('No dataset was deleted')


if __name__ == '__main__':
    parser = bmm.DatasetInitArgParser(description='find datasets')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='delete found datasets')
    parser.add_argument('--train', action='store_true',
                        help='only scan train subdir')
    parser.add_argument('--val', action='store_true',
                        help='only scan val subdir')
    parser.add_argument('--test', action='store_true',
                        help='only scan test subdir')
    filter_args, extra_args = parser.parse_args()
    main(extra_args, **vars(filter_args))
