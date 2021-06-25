import os


from brever.config import defaults
import brever.modelmanagement as bmm


def main(**kwargs):
    processed_dir = defaults().PATH.PROCESSED
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


if __name__ == '__main__':
    parser = bmm.DatasetInitArgParser(description='find datasets')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='delete found models')
    filter_args, extra_args = parser.parse_args()
    main(**vars(filter_args))
