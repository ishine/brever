import os
import shutil
import itertools

from brever.config import defaults
import brever.modelmanagement as bmm


def check_if_path_exists(configs, path_type, def_cfg):
    if path_type not in ['train', 'val']:
        raise ValueError('path_type must be train or val')
    default_path = bmm.get_config_field(def_cfg, f'{path_type}_path')
    for config in configs:
        path = bmm.get_config_field(config, f'{path_type}_path')
        if path is None:
            path = default_path
            msg = f'No {path_type} path specified, and default path does not '\
                  'exist'
        else:
            msg = f'The specified {path_type} path does not exist'
        if not os.path.exists(path):
            print(msg)
            return ask_user_yes_no('Do you wish to continue? y/n')
    return True


def check_if_test_datasets_exist(configs, def_cfg):
    default_paths = bmm.get_config_field(def_cfg, 'test_path')
    for config in configs:
        paths = bmm.get_config_field(config, 'test_path')
        if paths is None:
            paths = default_paths
            msg = 'No test paths specified, and not all the default test '\
                  'paths exist'
        else:
            msg = 'The specified test paths do not all exist'
        paths = bmm.globbed(paths)
        if not paths or any(not os.path.exists(path) for path in paths):
            print(msg)
            return ask_user_yes_no('Do you wish to continue? y/n')
    return True


def check_paths(configs, def_cfg):
    return (
        check_if_path_exists(configs, 'train', def_cfg)
        and check_if_path_exists(configs, 'val', def_cfg)
        and check_if_test_datasets_exist(configs, def_cfg)
    )


def ask_user_yes_no(msg):
    resp = None
    while resp not in ['y', 'Y', 'n', 'N']:
        resp = input(msg)
        if resp in ['y', 'Y']:
            return True
        elif resp in ['n', 'N']:
            return False
        else:
            print('Could not interpret answer')


def find_dset(
            dsets=None,
            configs=None,
            kind=None,
            speakers={'timit_.*'},
            rooms={'surrey_.*'},
            snr_dist_args=[-5, 10],
            target_angle_lims=[0.0, 0.0],
            noise_types={'dcase_.*'},
            random_rms=False,
        ):
    target_angle_min, target_angle_max = target_angle_lims
    return bmm.find_dataset(
        dsets=dsets,
        configs=configs,
        kind=kind,
        speakers=speakers,
        rooms=rooms,
        snr_dist_args=snr_dist_args,
        target_angle_min=target_angle_min,
        target_angle_max=target_angle_max,
        noise_types=noise_types,
        random_rms=random_rms,
    )


def main(args):
    angle_limss = [
        [0.0, 0.0],
        [-90.0, 90.0],
    ]
    snr_limss = [
        [-5, -5],
        [0, 0],
        [5, 5],
        [10, 10],
        [-5, 10],
    ]
    rms_jitters = [
        False,
        True,
    ]
    train_paths = []
    val_paths = []
    for a in angle_limss:
        dset, = find_dset(
            kind='train',
            target_angle_lims=a,
        )
        if dset not in train_paths:
            train_paths.append(dset)
            dset = dset.replace('train', 'val')
            val_paths.append(dset)
    for s in snr_limss:
        dset, = find_dset(
            kind='train',
            snr_dist_args=s,
        )
        if dset not in train_paths:
            train_paths.append(dset)
            dset = dset.replace('train', 'val')
            val_paths.append(dset)
    for r in rms_jitters:
        dset, = find_dset(
            kind='train',
            random_rms=r,
        )
        if dset not in train_paths:
            train_paths.append(dset)
            dset = dset.replace('train', 'val')
            val_paths.append(dset)

    args.train_path = train_paths
    args.val_path = val_paths
    args.test_path = [{'data/processed/test/*'}]
    args.layers = [2]
    args.hidden_sizes = [[1024, 1024]]
    args.stacks = [5]
    args.dropout = [True]
    args.seed = [1, 2, 3, 4]

    # ask if all train and val path combinations should be done
    combine_paths = True
    if args.train_path is not None and args.val_path is not None:
        n_train = len(args.train_path)
        n_val = len(args.val_path)
        if n_train == n_val and n_train != 1:
            n_combis = n_train*n_val
            msg = ('Same number of train and val paths detected. Do you wish '
                   'to initialize all pair combinations, or only as many '
                   f'pairs as paths specified? [Y to do all {n_combis} '
                   f'combinations / N to do only {n_train} pairs]')
            combine_paths = ask_user_yes_no(msg)

    # create dict of parameters to combine
    to_combine = {}
    for key in bmm.ModelFilterArgParser.arg_to_keys_map.keys():
        if key in ['train_path', 'val_path'] and not combine_paths:
            continue
        value = args.__getattribute__(key)
        if value is not None:
            bmm.set_config_field(to_combine, key, value)
    if not combine_paths:
        bmm.set_dict_field(to_combine, ['path_index'], list(range(n_train)))

    # make combinations
    if to_combine:
        to_combine = bmm.flatten(to_combine)
        keys, values = zip(*to_combine.items())
        configs = bmm.unflatten(keys, itertools.product(*values))
    else:
        configs = [{}]

    # add train and val paths if user requested to not make all combinations
    if not combine_paths:
        for config in configs:
            path_index = config['path_index']
            train_path = args.train_path[path_index]
            val_path = args.val_path[path_index]
            bmm.set_config_field(config, 'train_path', train_path)
            bmm.set_config_field(config, 'val_path', val_path)
            config.pop('path_index')

    def_cfg = defaults()
    models_dir = def_cfg.PATH.MODELS

    # check if paths exist
    result = check_paths(configs, def_cfg.to_dict())
    if not result:
        print('Aborting')
        return

    new_configs = []
    skipped = 0
    dupes = 0
    exists = 0

    for config in configs:
        def_cfg.update(config)  # throws an error if config is not valid

        model_id = bmm.get_unique_id(config)
        model_dir = os.path.join(models_dir, model_id)

        if os.path.exists(model_dir):
            exists += 1
            continue

        # exclude configs with uniform normalization features not included
        # in the list of features
        uni_feats = bmm.get_config_field(config, 'uni_norm_features', None)
        features = bmm.get_config_field(config, 'features', None)
        if (uni_feats is not None and features is not None
                and not uni_feats.issubset(features)):
            skipped += 1
            continue

        if config not in new_configs:
            new_configs.append(config)
        else:
            dupes += 1

    print(f'{len(configs)-skipped} config(s) attempted to be initialized.')
    print(f'{exists} already exist.')
    print(f'{dupes} are duplicates.')

    if not new_configs:
        print(f'{len(new_configs)} will be initialized.')
    else:
        msg = f'{len(new_configs)} will be initialized. Continue? y/n'
        proceed = ask_user_yes_no(msg)
        if proceed:
            for config in new_configs:
                unique_id = bmm.get_unique_id(config)
                dirpath = os.path.join(models_dir, unique_id)
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
                os.makedirs(dirpath)
                bmm.dump_yaml(config, os.path.join(dirpath, 'config.yaml'))
                print(f'Initialized {unique_id}')
        else:
            print('No model was initialized')


if __name__ == '__main__':
    parser = bmm.ModelFilterArgParser(description='initialize models')
    args, _ = parser.parse_args()
    main(args)
