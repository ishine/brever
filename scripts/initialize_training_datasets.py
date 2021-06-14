import os

import yaml

from brever.modelmanagement import set_dict_field, DatasetInitArgParser
from brever.config import defaults


def main(alias, params):

    processed_dir = defaults().PATH.PROCESSED

    for basename, filelims, number, seed in [
                ('training', [0.0, 0.7], args.n_train, args.seed_value_train),
                ('validation', [0.7, 0.85], args.n_val, args.seed_value_val),
            ]:
        config = {
            'PRE': {
                'SEED': {
                    'ON': True,
                    'VALUE': seed,
                },
                'MIX': {
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
                key_list = DatasetInitArgParser.arg_to_keys_map[param]
                set_dict_field(config, key_list, value)

        dirname = f'{basename}_{args.alias}'
        dirpath = os.path.join(processed_dir, dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        config_filepath = os.path.join(dirpath, 'config.yaml')
        if os.path.exists(config_filepath) and not args.force:
            print(f'{config_filepath} already exists')
            continue
        with open(config_filepath, 'w') as f:
            yaml.dump(config, f)
        print(f'Created {config_filepath}')


if __name__ == '__main__':
    parser = DatasetInitArgParser(description='initialize training datasets')
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
