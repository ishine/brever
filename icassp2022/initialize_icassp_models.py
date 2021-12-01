import os
import shutil
# import itertools
import copy

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
            target_angle_lims=[-90, 90.0],
            noise_types={'dcase_.*'},
            random_rms=False,
            filelims_room=None,
            features={'logfbe'}
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
        filelims_room=filelims_room,
        features={'logfbe'}
    )


def add_config(configs, seed, train_path, val_path, test_paths, layers=2,
               hidden_sizes=[1024, 1024], stacks=5, dropout=True, args=None):
    config = {}
    bmm.set_config_field(config, 'layers', layers)
    bmm.set_config_field(config, 'hidden_sizes', hidden_sizes)
    bmm.set_config_field(config, 'stacks', stacks)
    bmm.set_config_field(config, 'dropout', dropout)
    bmm.set_config_field(config, 'seed', seed)
    bmm.set_config_field(config, 'train_path', train_path)
    bmm.set_config_field(config, 'val_path', val_path)
    bmm.set_config_field(config, 'test_path', test_paths)
    if args is not None:
        for key, vals in args.__dict__.items():
            if vals is not None:
                if len(vals) > 1:
                    raise ValueError('only one value per hyperparameter is '
                                     'allowed')
                val, = vals
                bmm.set_config_field(config, key, val)
    configs.append(config)


def main(args):

    train_dsets, train_configs = bmm.find_dataset('train', return_configs=True)
    test_dsets, test_configs = bmm.find_dataset('test', return_configs=True)

    configs = []

    # inner corpus
    dict_ = {
        'speakers': [
            {
                'dbase': 'timit',
                'types': [
                    'm0',
                    'f0',
                    'm1',
                    'f1',
                    'm2',
                    'f2',
                ],
            },
            {
                'dbase': 'timit',
                'types': [
                    '(f[0-4]|m[0-4])',
                    '(f[5-9]|m[5-9])',
                    '(f1[5-9]|m1[5-9])',
                    '(f1[0-4]|m1[0-4])',
                    '(f2[0-4]|m2[0-4])',
                ],
            },
            {
                'dbase': 'timit',
                'types': [
                    '(f[0-4]?[0-9]|m[0-4]?[0-9])',
                    '(f[4-9][0-9]|m[4-9][0-9])',
                    '(f1[0-4][0-9]|m1[0-4][0-9])',
                    '(f[0-9]?[02468]|m[0-9]?[02468])',
                    '(f[0-9]?[13579]|m[0-9]?[13579])',
                ],
            },
            {
                'dbase': 'libri',
                'types': [
                    'm0',
                    'f0',
                    'm1',
                    'f1',
                    'm2',
                    'f2',
                ],
            },
            {
                'dbase': 'libri',
                'types': [
                    '(f[0-4]|m[0-4])',
                    '(f[5-9]|m[5-9])',
                    '(f1[0-4]|m1[0-4])',
                    '(f1[5-9]|m1[5-9])',
                    '(f2[0-4]|m2[0-4])',
                ],
            },
            {
                'dbase': 'libri',
                'types': [
                    '(f[0-4]?[0-9]|m[0-4]?[0-9])',
                    '(f[4-9][0-9]|m[4-9][0-9])',
                    '(f[0-9]?[02468]|m[0-9]?[02468])',
                    '(f[0-9]?[13579]|m[0-9]?[13579])',
                ],
            },
        ],
        'rooms': [
            {
                'dbase': 'surrey',
                'types': [
                    'anechoic',
                    'room_a',
                    'room_b',
                    'room_c',
                    'room_d',
                ],
            },
            {
                'dbase': 'ash',
                'types': [
                    'r01',
                    'r02',
                    'r03',
                    'r04',
                    'r05a?b?',
                ],
            },
            {
                'dbase': 'ash',
                'types': [
                    'r0[0-9]a?b?',
                    'r1[0-9]',
                    'r2[0-9]',
                    'r3[0-9]',
                    'r(00|04|08|12|16|20|24|18|32|36)',
                ],
            },
        ],
        'noise_types': [
            {
                'dbase': 'dcase',
                'types': [
                    'airport',
                    'bus',
                    'metro',
                    'metro_station',
                    'park',
                ],
            },
            {
                'dbase': 'noisex',
                'types': [
                    'babble',
                    'buccaneer1',
                    'destroyerengine',
                    'f16',
                    'factory1',
                ],
            },
        ],
    }

    for dim, experiments in dict_.items():
        for exp in experiments:
            dbase, types = exp['dbase'], exp['types']
            test_paths = set()
            for type_ in types:
                kwargs_list = [
                    {dim: set([f'{dbase}_(?!{type_}$).*'])},
                ]
                if dbase in ['dcase', 'noisex', 'surrey']:
                    kwargs_list.append(
                        {dim: set([f'{dbase}_{type_}'])},
                    )
                for kwargs in kwargs_list:
                    test_path, = find_dset(
                        dsets=test_dsets,
                        configs=test_configs,
                        filelims_room='odd',
                        **kwargs,
                    )
                    test_paths.add(test_path)
            for type_ in types:
                kwargs_list = [
                    {dim: set([f'{dbase}_{type_}'])},
                    {dim: set([f'{dbase}_(?!{type_}$).*'])},
                ]
                for kwargs in kwargs_list:
                    train_path, = find_dset(
                        dsets=train_dsets,
                        configs=train_configs,
                        filelims_room='even',
                        **kwargs,
                    )
                    val_path = train_path.replace('train', 'val')
                    add_config(configs, 0, train_path, val_path, test_paths)

    # cross corpus
    dict_ = {
        'speakers': [
            'timit_.*',
            'libri_.*',
            'ieee',
            'arctic',
            'hint',
        ],
        'noise_types': [
            'dcase_.*',
            'noisex_.*',
            'icra_.*',
            'demand',
            'arte',
        ],
        'rooms': [
            'surrey_.*',
            'ash_.*',
            'bras_.*',
            'catt_.*',
            'avil_.*',
        ],
    }
    # single, double and triple mismatch
    from itertools import combinations
    dims_ = [x for n in range(4) for x in combinations(dict_.keys(), n)]
    for dims in dims_:
        test_paths = set()
        for dbases in zip(*[dict_[dim] for dim in dims]):
            kwargs = {dim: set([dbase]) for dim, dbase in zip(dims, dbases)}
            test_path, = find_dset(
                dsets=test_dsets,
                configs=test_configs,
                filelims_room='odd',
                **kwargs,
            )
            test_paths.add(test_path)
        for dbases in zip(*[dict_[dim] for dim in dims]):
            kwargs_list = [
                {dim: set([dbase]) for dim, dbase in zip(dims, dbases)},
                {dim: set([db for db in dict_[dim] if db != dbase])
                 for dim, dbase in zip(dims, dbases)},
            ]
            for kwargs in kwargs_list:
                train_path, = find_dset(
                    dsets=train_dsets,
                    configs=train_configs,
                    filelims_room='even',
                    **kwargs,
                )
                val_path = train_path.replace('train', 'val')
                add_config(configs, 0, train_path, val_path, test_paths)
                # for the triple mismatch, add extra models specified in the
                # command line arguments
                if len(dims) == 3:
                    add_config(configs, 0, train_path, val_path, test_paths,
                               args=args)
                # also add pdf and logpdf datasets
                for features in ['pdf', 'logpdf']:
                    train_path, = find_dset(
                        dsets=train_dsets,
                        configs=train_configs,
                        filelims_room='even',
                        **kwargs,
                        features=features,
                    )
                    val_path = train_path.replace('train', 'val')
                    add_config(configs, 0, train_path, val_path, test_paths)

    # snr, direction and level experiments
    dict_ = {
        'target_angle_lims': [
            [0.0, 0.0],
            [-90.0, 90.0],
        ],
        'snr_dist_args': [
            [-5, -5],
            [0, 0],
            [5, 5],
            [10, 10],
            [-5, 10],
        ],
        'random_rms': [
            False,
            True,
        ],
    }
    for dim, values in dict_.items():
        if dim == 'target_angle_lims':
            train_rooms = 'all'
            test_rooms = 'all'
        else:
            train_rooms = 'even'
            test_rooms = 'odd'
        test_paths = set()
        for val in values:
            kwargs = {dim: val}
            test_path, = find_dset(
                dsets=test_dsets,
                configs=test_configs,
                filelims_room=test_rooms,
                **kwargs,
            )
            test_paths.add(test_path)
        for val in values:
            kwargs = {dim: val}
            train_path, = find_dset(
                dsets=train_dsets,
                configs=train_configs,
                filelims_room=train_rooms,
                **kwargs,
            )
            val_path = train_path.replace('train', 'val')
            for seed in range(5):
                add_config(configs, seed, train_path, val_path, test_paths)

    # merge test paths of models with the same train path
    for i, config_1 in enumerate(configs):
        for j, config_2 in enumerate(configs):
            if j > i:
                train_1 = bmm.get_config_field(config_1, 'train_path')
                train_2 = bmm.get_config_field(config_2, 'train_path')
                seed_1 = bmm.get_config_field(config_1, 'seed')
                seed_2 = bmm.get_config_field(config_2, 'seed')
                if train_1 == train_2 and seed_1 == seed_2:
                    test_path_1 = bmm.get_config_field(
                        config_1,
                        'test_path',
                    )
                    test_path_2 = bmm.get_config_field(
                        config_2,
                        'test_path',
                    )
                    bmm.set_config_field(
                        config_1,
                        'test_path',
                        test_path_1 | test_path_2,
                    )
                    bmm.set_config_field(
                        config_2,
                        'test_path',
                        test_path_1 | test_path_2,
                    )

    def_cfg = defaults()
    models_dir = def_cfg.PATH.MODELS

    # check if paths exist
    result = check_paths(configs, def_cfg.to_dict())
    if not result:
        print('Aborting')
        return

    # check for dupes
    temp = []
    for c in configs:
        if c not in temp:
            temp.append(c)
    dupes = len(configs) - len(temp)
    configs = temp
    del temp

    new_configs = []
    skipped = 0
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

        new_configs.append(config)

    # find existing models only differing by their test paths
    totally_new_configs = []
    exist_but_with_different_test_path = []
    existing_models = []
    existing_configs = []
    for model_id in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_id)
        cfg_path = os.path.join(model_dir, 'config.yaml')
        cfg = bmm.read_yaml(cfg_path)
        existing_models.append(model_dir)
        existing_configs.append(cfg)
    existing_tests = [c['POST']['PATH'].pop('TEST') for c in existing_configs]
    for config in new_configs:
        model_id = bmm.get_unique_id(config)
        model_dir = os.path.join(models_dir, model_id)
        if not(os.path.exists(model_dir)):
            copy_ = copy.deepcopy(config)
            copy_id = bmm.get_unique_id(config)
            new_tests = copy_['POST']['PATH'].pop('TEST')
            try:
                index = existing_configs.index(copy_)
            except ValueError:
                totally_new_configs.append(config)
            else:
                exist_but_with_different_test_path.append((
                    os.path.join(models_dir, copy_id),
                    new_tests,
                    existing_models[index],
                    existing_tests[index]
                ))

    # if such models exists, ask user what to do with them
    if exist_but_with_different_test_path:
        print(f'{len(exist_but_with_different_test_path)} models were '
              'attempted to be initialized but already exist using different '
              'test paths')
        msg = 'Would you like to add the test paths to the old list of ' \
              'paths intead? [overwrite/merge/new]'
        resp = None
        while resp not in ['overwrite', 'merge', 'new']:
            resp = input(msg)
            if resp.lower() == 'overwrite':
                print("Overwriting. It is recommended to run "
                      "'python scripts/check_sanity.py' afterwards to rename "
                      "all the models")
                for items in exist_but_with_different_test_path:
                    model, tests, old_model, old_tests = items
                    cfg_path = os.path.join(old_model, 'config.yaml')
                    cfg = bmm.read_yaml(cfg_path)
                    bmm.set_config_field(cfg, 'test_path', tests)
                    bmm.dump_yaml(cfg, cfg_path)
                new_configs = totally_new_configs
                break
            elif resp.lower() == 'merge':
                print("Merging. It is recommended to run "
                      "'python scripts/check_sanity.py' afterwards to rename "
                      "all the models")
                for items in exist_but_with_different_test_path:
                    model, tests, old_model, old_tests = items
                    cfg_path = os.path.join(old_model, 'config.yaml')
                    cfg = bmm.read_yaml(cfg_path)
                    bmm.set_config_field(cfg, 'test_path', tests | old_tests)
                    bmm.dump_yaml(cfg, cfg_path)
                new_configs = totally_new_configs
                break
            elif resp.lower() == 'new':
                break
            else:
                print('Could not interpret answer')

    print(f'{len(configs)-skipped+dupes} config(s) attempted to be '
          'initialized.')
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
