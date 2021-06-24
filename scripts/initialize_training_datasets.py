import os

import brever.modelmanagement as bmm
from brever.config import defaults


def main(args, params):

    processed_dir = defaults().PATH.PROCESSED

    for basename, filelims, number, seed in [
                ('train', [0.0, 0.7], args.n_train, args.seed_value_train),
                ('val', [0.7, 0.85], args.n_val, args.seed_value_val),
            ]:
        config = {
            'PRE': {
                'SEED': {
                    'ON': True,
                    'VALUE': seed,
                },
                'MIX': {
                    'SAVE': False,
                    'NUMBER': number,
                    'FILELIMITS': {
                        'NOISE': filelims.copy(),
                        'TARGET': filelims.copy(),
                    },
                }
            }
        }

        for param, value in params.items():
            if value is not None:
                key_list = bmm.DatasetInitArgParser.arg_to_keys_map[param]
                bmm.set_dict_field(config, key_list, value)

        dirpath = os.path.join(processed_dir, basename, args.alias)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        config_filepath = os.path.join(dirpath, 'config.yaml')
        if os.path.exists(config_filepath) and not args.force:
            print(f'{config_filepath} already exists')
            continue
        bmm.dump_yaml(config, config_filepath)
        print(f'Created {config_filepath}')


if __name__ == '__main__':
    parser = bmm.DatasetInitArgParser(description='initialize train datasets')
    parser.add_argument('alias',
                        help='dataset alias')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('--n-train', type=int, default=10000,
                        help='number of training mixture, defaults to 1000')
    parser.add_argument('--n-val', type=int, default=2000,
                        help='number of validation mixture, defaults to 200')
    parser.add_argument('--seed-value-train', type=int, default=0,
                        help='seed for the training dataset')
    parser.add_argument('--seed-value-val', type=int, default=1,
                        help='seed for the validation dataset')
    dataset_args, args = parser.parse_args()
    params = vars(dataset_args)

    main(args, params)
