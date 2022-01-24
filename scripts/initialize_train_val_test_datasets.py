import os

from brever.config import defaults
import brever.management as bm


def make_config(kind, params):
    if kind == 'train':
        filelims = [0.0, 0.7]
        duration = 36000
        seed = 0
        save = False
        rooms = 'even'
    elif kind == 'val':
        filelims = [0.7, 0.85]
        duration = 100
        seed = 1
        save = False
        rooms = 'even'
    if kind == 'test':
        filelims = [0.7, 1.0]
        duration = 1800
        seed = 2
        save = True
        rooms = 'odd'

    config = {
        'PRE': {
            'SEED': {
                'ON': True,
                'VALUE': seed,
            },
            'MIX': {
                'SAVE': save,
                'TOTALDURATION': duration,
                'FILELIMITS': {
                    'NOISE': filelims.copy(),
                    'TARGET': filelims.copy(),
                    'ROOM': rooms,
                },
            }
        }
    }

    for param, value in params.items():
        if value is not None:
            key_list = bm.DatasetInitArgParser.arg_to_keys_map[param]
            bm.set_dict_field(config, key_list, value)

    return config


def main(args, params):
    def_cfg = defaults()
    processed_dir = def_cfg.PATH.PROCESSED

    for kind in ['train', 'val', 'test']:
        config = make_config(kind, params)
        def_cfg.update(config)  # throws an error if config is not valid

        if kind == 'val':
            dset_id = bm.get_unique_id(make_config('train', params))
        else:
            dset_id = bm.get_unique_id(config)
        dset_path = os.path.join(processed_dir, kind, dset_id)

        if not os.path.exists(dset_path):
            os.makedirs(dset_path)
        config_filepath = os.path.join(dset_path, 'config.yaml')
        if os.path.exists(config_filepath) and not args.force:
            print(f'Already exists! {config_filepath} ')
        else:
            bm.dump_yaml(config, config_filepath)
            print(f'Created {config_filepath}')


if __name__ == '__main__':
    parser = bm.DatasetInitArgParser(description='initialize dataset')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
