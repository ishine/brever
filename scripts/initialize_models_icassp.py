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

    train_dsets, train_configs = bmm.find_dataset('train', return_configs=True)
    test_dsets, test_configs = bmm.find_dataset('test', return_configs=True)

    experiments = [
        {
            'dim': 'speakers',
            'train': [
                {'timit_m0'},
                {'timit_f0'},
                {'timit_m1'},
                {'timit_f1'},
                {'timit_m2'},
                {'timit_f2'},
                {'timit_(?!m0$).*'},
                {'timit_(?!f0$).*'},
                {'timit_(?!m1$).*'},
                {'timit_(?!f1$).*'},
                {'timit_(?!m2$).*'},
                {'timit_(?!f2$).*'},
            ],
            'test': [
                {'timit_(?!m0$).*'},
                {'timit_(?!f0$).*'},
                {'timit_(?!m1$).*'},
                {'timit_(?!f1$).*'},
                {'timit_(?!m2$).*'},
                {'timit_(?!f2$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'speakers',
            'train': [
                {'timit_(f[0-4]|m[0-4])'},
                {'timit_(f[5-9]|m[5-9])'},
                {'timit_(f1[5-9]|m1[5-9])'},
                {'timit_(f1[0-4]|m1[0-4])'},
                {'timit_(f2[0-4]|m2[0-4])'},
                {'timit_(?!(f[0-4]|m[0-4])$).*'},
                {'timit_(?!(f[5-9]|m[5-9])$).*'},
                {'timit_(?!(f1[0-4]|m1[0-4])$).*'},
                {'timit_(?!(f1[5-9]|m1[5-9])$).*'},
                {'timit_(?!(f2[0-4]|m2[0-4])$).*'},
            ],
            'test': [
                {'timit_(?!(f[0-4]|m[0-4])$).*'},
                {'timit_(?!(f[5-9]|m[5-9])$).*'},
                {'timit_(?!(f1[0-4]|m1[0-4])$).*'},
                {'timit_(?!(f1[5-9]|m1[5-9])$).*'},
                {'timit_(?!(f2[0-4]|m2[0-4])$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'speakers',
            'train': [
                {'timit_(f[0-4]?[0-9]|m[0-4]?[0-9])'},
                {'timit_(f[4-9][0-9]|m[4-9][0-9])'},
                {'timit_(f1[0-4][0-9]|m1[0-4][0-9])'},
                {'timit_(f[0-9]?[02468]|m[0-9]?[02468])'},
                {'timit_(f[0-9]?[13579]|m[0-9]?[13579])'},
                {'timit_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},
                {'timit_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},
                {'timit_(?!(f1[0-4][0-9]|m1[0-4][0-9])$).*'},
                {'timit_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},
                {'timit_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},
                ],
            'test': [
                {'timit_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},
                {'timit_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},
                {'timit_(?!(f1[0-4][0-9]|m1[0-4][0-9])$).*'},
                {'timit_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},
                {'timit_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'speakers',
            'train': [
                {'libri_m0'},
                {'libri_f0'},
                {'libri_m1'},
                {'libri_f1'},
                {'libri_m2'},
                {'libri_f2'},
                {'libri_(?!m0$).*'},
                {'libri_(?!f0$).*'},
                {'libri_(?!m1$).*'},
                {'libri_(?!f1$).*'},
                {'libri_(?!m2$).*'},
                {'libri_(?!f2$).*'},
            ],
            'test': [
                {'libri_(?!m0$).*'},
                {'libri_(?!f0$).*'},
                {'libri_(?!m1$).*'},
                {'libri_(?!f1$).*'},
                {'libri_(?!m2$).*'},
                {'libri_(?!f2$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'speakers',
            'train': [
                {'libri_(f[0-4]|m[0-4])'},
                {'libri_(f[5-9]|m[5-9])'},
                {'libri_(f1[0-4]|m1[0-4])'},
                {'libri_(f1[5-9]|m1[5-9])'},
                {'libri_(f2[0-4]|m2[0-4])'},
                {'libri_(?!(f[0-4]|m[0-4])$).*'},
                {'libri_(?!(f[5-9]|m[5-9])$).*'},
                {'libri_(?!(f1[0-4]|m1[0-4])$).*'},
                {'libri_(?!(f1[5-9]|m1[5-9])$).*'},
                {'libri_(?!(f2[0-4]|m2[0-4])$).*'},
            ],
            'test': [
                {'libri_(?!(f[0-4]|m[0-4])$).*'},
                {'libri_(?!(f[5-9]|m[5-9])$).*'},
                {'libri_(?!(f1[0-4]|m1[0-4])$).*'},
                {'libri_(?!(f1[5-9]|m1[5-9])$).*'},
                {'libri_(?!(f2[0-4]|m2[0-4])$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'speakers',
            'train': [
                {'libri_(f[0-4]?[0-9]|m[0-4]?[0-9])'},
                {'libri_(f[4-9][0-9]|m[4-9][0-9])'},
                {'libri_(f[0-9]?[02468]|m[0-9]?[02468])'},
                {'libri_(f[0-9]?[13579]|m[0-9]?[13579])'},
                {'libri_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},
                {'libri_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},
                {'libri_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},
                {'libri_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},
            ],
            'test': [
                {'libri_(?!(f[0-4]?[0-9]|m[0-4]?[0-9])$).*'},
                {'libri_(?!(f[4-9][0-9]|m[4-9][0-9])$).*'},
                {'libri_(?!(f[0-9]?[02468]|m[0-9]?[02468])$).*'},
                {'libri_(?!(f[0-9]?[13579]|m[0-9]?[13579])$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'speakers',
            'train': [
                {'ieee'},
                {'timit_.*'},
                {'libri_.*'},
                {'arctic'},
                {'hint'},
                {'libri_.*', 'timit_.*', 'arctic', 'hint'},
                {'ieee', 'timit_.*', 'arctic', 'hint'},
                {'ieee', 'libri_.*', 'arctic', 'hint'},
                {'ieee', 'libri_.*', 'timit_.*', 'hint'},
                {'ieee', 'libri_.*', 'timit_.*', 'arctic'},
            ],
            'test': [
                {'ieee'},
                {'timit_.*'},
                {'libri_.*'},
                {'arctic'},
                {'hint'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'rooms',
            'train': [
                {'surrey_anechoic'},
                {'surrey_room_a'},
                {'surrey_room_b'},
                {'surrey_room_c'},
                {'surrey_room_d'},
                {'surrey_(?!anechoic$).*'},
                {'surrey_(?!room_a$).*'},
                {'surrey_(?!room_b$).*'},
                {'surrey_(?!room_c$).*'},
                {'surrey_(?!room_d$).*'},
            ],
            'test': [
                {'surrey_(?!anechoic$).*'},
                {'surrey_(?!room_a$).*'},
                {'surrey_(?!room_b$).*'},
                {'surrey_(?!room_c$).*'},
                {'surrey_(?!room_d$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'rooms',
            'train': [
                {'ash_r01'},
                {'ash_r02'},
                {'ash_r03'},
                {'ash_r04'},
                {'ash_r05a?b?'},
                {'ash_(?!r01$).*'},
                {'ash_(?!r02$).*'},
                {'ash_(?!r03$).*'},
                {'ash_(?!r04$).*'},
                {'ash_(?!r05a?b?$).*'},
            ],
            'test': [
                {'ash_(?!r01$).*'},
                {'ash_(?!r02$).*'},
                {'ash_(?!r03$).*'},
                {'ash_(?!r04$).*'},
                {'ash_(?!r05a?b?$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'rooms',
            'train': [
                {'ash_r0[0-9]a?b?'},
                {'ash_r1[0-9]'},
                {'ash_r2[0-9]'},
                {'ash_r3[0-9]'},
                {'ash_r(00|04|08|12|16|20|24|18|32|36)'},
                {'ash_(?!r0[0-9]a?b?$).*'},
                {'ash_(?!r1[0-9]$).*'},
                {'ash_(?!r2[0-9]$).*'},
                {'ash_(?!r3[0-9]$).*'},
                {'ash_(?!r(00|04|08|12|16|20|24|18|32|36)$).*'},
            ],
            'test': [
                {'ash_(?!r0[0-9]a?b?$).*'},
                {'ash_(?!r1[0-9]$).*'},
                {'ash_(?!r2[0-9]$).*'},
                {'ash_(?!r3[0-9]$).*'},
                {'ash_(?!r(00|04|08|12|16|20|24|18|32|36)$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'rooms',
            'train': [
                {'surrey_.*'},
                {'ash_.*'},
                {'elospheres_.*'},
                {'catt_.*'},
                {'avil_.*'},
                {'ash_.*', 'elospheres_.*', 'catt_.*', 'avil_.*'},
                {'surrey_.*', 'elospheres_.*', 'catt_.*', 'avil_.*'},
                {'surrey_.*', 'ash_.*', 'catt_.*', 'avil_.*'},
                {'surrey_.*', 'ash_.*', 'elospheres_.*', 'avil_.*'},
                {'surrey_.*', 'ash_.*', 'elospheres_.*', 'catt_.*'},
            ],
            'test': [
                {'surrey_.*'},
                {'ash_.*'},
                {'elospheres_.*'},
                {'catt_.*'},
                {'avil_.*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'noise_types',
            'train': [
                {'dcase_airport'},
                {'dcase_bus'},
                {'dcase_metro'},
                {'dcase_metro_station'},
                {'dcase_park'},
                {'dcase_(?!airport$).*'},
                {'dcase_(?!bus$).*'},
                {'dcase_(?!metro$).*'},
                {'dcase_(?!metro_station$).*'},
                {'dcase_(?!park$).*'},
            ],
            'test': [
                {'dcase_(?!airport$).*'},
                {'dcase_(?!bus$).*'},
                {'dcase_(?!metro$).*'},
                {'dcase_(?!metro_station$).*'},
                {'dcase_(?!park$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'noise_types',
            'train': [
                {'noisex_babble'},
                {'noisex_buccaneer1'},
                {'noisex_destroyerengine'},
                {'noisex_f16'},
                {'noisex_factory1'},
                {'noisex_(?!babble$).*'},
                {'noisex_(?!buccaneer1$).*'},
                {'noisex_(?!destroyerengine$).*'},
                {'noisex_(?!f16$).*'},
                {'noisex_(?!factory1$).*'},
            ],
            'test': [
                {'noisex_(?!babble$).*'},
                {'noisex_(?!buccaneer1$).*'},
                {'noisex_(?!destroyerengine$).*'},
                {'noisex_(?!f16$).*'},
                {'noisex_(?!factory1$).*'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'noise_types',
            'train': [
                {'dcase_.*'},
                {'icra_.*'},
                {'noisex_.*'},
                {'demand'},
                {'arte'},
                {'icra_.*', 'demand', 'noisex_.*', 'arte'},
                {'dcase_.*', 'demand', 'noisex_.*', 'arte'},
                {'dcase_.*', 'icra_.*', 'noisex_.*', 'arte'},
                {'dcase_.*', 'icra_.*', 'demand', 'arte'},
                {'dcase_.*', 'icra_.*', 'demand', 'noisex_.*'},
            ],
            'test': [
                {'dcase_.*'},
                {'icra_.*'},
                {'noisex_.*'},
                {'demand'},
                {'arte'},
            ],
            'seeds': [0],
        },
        {
            'dim': 'snr_dist_args',
            'train': [
                [-5, -5],
                [0, 0],
                [5, 5],
                [10, 10],
                [-5, 10],
            ],
            'test': [
                [-5, -5],
                [0, 0],
                [5, 5],
                [10, 10],
                [-5, 10],
            ],
            'seeds': [0, 1, 2, 3, 4],
        },
        {
            'dim': 'target_angle_lims',
            'train': [
                [0.0, 0.0],
                [-90.0, 90.0],
            ],
            'test': [
                [0.0, 0.0],
                [-90.0, 90.0],
            ],
            'seeds': [0, 1, 2, 3, 4],
        },
        {
            'dim': 'random_rms',
            'train': [
                False,
                True,
            ],
            'test': [
                False,
                True,
            ],
            'seeds': [0, 1, 2, 3, 4],
        },
    ]

    configs = []
    for exp in experiments:
        test_paths = set()
        for x in exp['test']:
            dset, = find_dset(
                dsets=test_dsets,
                configs=test_configs,
                **{exp['dim']: x},
            )
            test_paths.add(dset)
        for x in exp['train']:
            dset, = find_dset(
                dsets=train_dsets,
                configs=train_configs,
                **{exp['dim']: x},
            )
            train_path = dset
            val_path = dset.replace('train', 'val')
            for seed in exp['seeds']:
                config = {}
                bmm.set_config_field(config, 'layers', 2)
                bmm.set_config_field(config, 'hidden_sizes', [1024, 1024])
                bmm.set_config_field(config, 'stacks', 5)
                bmm.set_config_field(config, 'dropout', True)
                bmm.set_config_field(config, 'seed', seed)
                bmm.set_config_field(config, 'train_path', train_path)
                bmm.set_config_field(config, 'val_path', val_path)
                bmm.set_config_field(config, 'test_path', test_paths)
                configs.append(config)

    # deal with the default model which appeared in both lists before merging
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
