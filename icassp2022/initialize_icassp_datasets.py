import os
import argparse

import brever.modelmanagement as bmm
from brever.config import defaults


def main():
    def_cfg = defaults()
    processed_dir = def_cfg.PATH.PROCESSED

    # the default config is defined by the default arguments
    def add_config(
                configs,
                kind,
                noise_types={'dcase_.*'},
                rooms={'surrey_.*'},
                angle_lims=[-90.0, 90.0],
                snr_lims=[-5, 10],
                rms_jitter=False,
                speakers={'timit_.*'},
                filelims_rooms=None,
                features=None,
            ):

        def make_config(kind_):
            if kind_ == 'train':
                filelims = [0.0, 0.7]
                duration = 36000
                seed = 0
                save = False
                filelims_rooms_ = filelims_rooms or 'even'
            elif kind_ == 'val':
                filelims = [0.7, 0.85]
                duration = 36000
                seed = 1
                save = False
                filelims_rooms_ = filelims_rooms or 'even'
            if kind_ == 'test':
                filelims = [0.7, 1.0]
                duration = 1800
                seed = 2
                save = True
                filelims_rooms_ = filelims_rooms or 'odd'

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
                            'ROOM': filelims_rooms_,
                        },
                        'RANDOM': {
                            'ROOMS': rooms,
                            'TARGET': {
                                'ANGLE': {
                                    'MIN': angle_lims[0],
                                    'MAX': angle_lims[1]
                                },
                                'SNR': {
                                    'DISTARGS': [snr_lims[0], snr_lims[1]],
                                    'DISTNAME': 'uniform'
                                },
                                'SPEAKERS': speakers
                            },
                            'SOURCES': {
                                    'TYPES': noise_types
                            },
                            'RMSDB': {
                                'ON': rms_jitter
                            }
                        }
                    }
                }
            }

            if features is not None:
                bmm.set_config_field(config, 'features', features)

            return config

        config = make_config(kind)
        def_cfg.update(config)  # throws an error if config is not valid

        if kind == 'val':
            dset_id = bmm.get_unique_id(make_config('train'))
        else:
            dset_id = bmm.get_unique_id(config)
        dset_path = os.path.join(processed_dir, kind, dset_id)

        if (config, dset_path) not in configs:
            configs.append((config, dset_path))

    configs = []

    # inner corpus
    dict_ = {
        'speakers': {
            'timit': [
                'm0',
                'f0',
                'm1',
                'f1',
                'm2',
                'f2',
                '(f[0-4]|m[0-4])',
                '(f[5-9]|m[5-9])',
                '(f1[0-4]|m1[0-4])',
                '(f1[5-9]|m1[5-9])',
                '(f2[0-4]|m2[0-4])',
                '(f[0-4]?[0-9]|m[0-4]?[0-9])',
                '(f[4-9][0-9]|m[4-9][0-9])',
                '(f1[0-4][0-9]|m1[0-4][0-9])',
                '(f[0-9]?[02468]|m[0-9]?[02468])',
                '(f[0-9]?[13579]|m[0-9]?[13579])',
            ],
            'libri': [
                'm0',
                'f0',
                'm1',
                'f1',
                'm2',
                'f2',
                '(f[0-4]|m[0-4])',
                '(f[5-9]|m[5-9])',
                '(f1[0-4]|m1[0-4])',
                '(f1[5-9]|m1[5-9])',
                '(f2[0-4]|m2[0-4])',
                '(f[0-4]?[0-9]|m[0-4]?[0-9])',
                '(f[4-9][0-9]|m[4-9][0-9])',
                '(f[0-9]?[02468]|m[0-9]?[02468])',
                '(f[0-9]?[13579]|m[0-9]?[13579])',
            ],
        },
        'noise_types': {
            'dcase': [
                'airport',
                'bus',
                'metro',
                'metro_station',
                'park',
            ],
            'noisex': [
                'babble',
                'buccaneer1',
                'destroyerengine',
                'f16',
                'factory1',
            ],
        },
        'rooms': {
            'surrey': [
                'anechoic',
                'room_a',
                'room_b',
                'room_c',
                'room_d',
            ],
            'ash': [
                'r01',
                'r02',
                'r03',
                'r04',
                'r05a?b?',
                'r0[0-9]a?b?',
                'r1[0-9]',
                'r2[0-9]',
                'r3[0-9]',
                'r(00|04|08|12|16|20|24|18|32|36)',
            ],
        },
    }
    for dim, subdict in dict_.items():
        for dbase, types in subdict.items():
            for type_ in types:
                kwargs = {dim: set([f'{dbase}_{type_}'])}
                add_config(configs, 'train', **kwargs)
                add_config(configs, 'val', **kwargs)
                if dbase in ['dcase', 'noisex', 'surrey']:
                    add_config(configs, 'test', **kwargs)
                kwargs = {dim: set([f'{dbase}_(?!{type_}$).*'])}
                add_config(configs, 'train', **kwargs)
                add_config(configs, 'val', **kwargs)
                add_config(configs, 'test', **kwargs)

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

    def routine(dims, dbases, features=None):
        kwargs = {dim: set([dbase]) for dim, dbase in zip(dims, dbases)}
        add_config(configs, 'train', **kwargs, features=features)
        add_config(configs, 'val', **kwargs, features=features)
        add_config(configs, 'test', **kwargs, features=features)
        kwargs = {dim: set([db for db in dict_[dim] if db != dbase])
                  for dim, dbase in zip(dims, dbases)}
        add_config(configs, 'train', **kwargs, features=features)
        add_config(configs, 'val', **kwargs, features=features)

    dims_ = [x for n in range(4) for x in combinations(dict_.keys(), n)]
    for dims in dims_:
        for dbases in zip(*[dict_[dim] for dim in dims]):
            routine(dims, dbases)
            # for the triple mismatch, add pdf and logpdf datasets
            if len(dims) == 3:
                routine(dims, dbases, features={'pdf'})
                routine(dims, dbases, features={'logpdf'})

    # SNR, direction and level dimensions
    dict_ = {
        'angle_lims': [
            [0.0, 0.0],
            [-90.0, 90.0],
        ],
        'snr_lims': [
            [-5, -5],
            [0, 0],
            [5, 5],
            [10, 10],
            [-5, 10],
        ],
        'rms_jitter': [
            False,
            True,
        ],
    }
    for dim, values in dict_.items():
        for val in values:
            kwargs = {dim: val}
            if dim == 'angle_lims':
                kwargs['filelims_rooms'] = 'all'
            add_config(configs, 'train', **kwargs)
            add_config(configs, 'val', **kwargs)
            add_config(configs, 'test', **kwargs)

    new_configs = []
    for config_dict, dset_path in configs:
        if not os.path.exists(dset_path):
            new_configs.append((config_dict, dset_path))

    print(f'{len(configs)} datasets attempted to be initialized.')
    print(f'{len(configs) - len(new_configs)} already exist.')

    # build the list of dsets already in the filesystem
    filesystem_dsets = []
    for subdir in ['train', 'val', 'test']:
        for file in os.listdir(os.path.join(processed_dir, subdir)):
            filesystem_dsets.append(os.path.join(processed_dir, subdir, file))
    # highlight the dsets in the filesystem that were not attempted to be
    # created again; they might be deprecated
    deprecated_dsets = []
    for dset in filesystem_dsets:
        if dset not in [config[1] for config in configs]:
            deprecated_dsets.append(dset)
    if deprecated_dsets:
        print('The following datasets are in the filesystem but were not '
              'attempted to be initialized again. They might be deprecated?')
        for dset in deprecated_dsets:
            print(dset)

    if not new_configs:
        print(f'{len(new_configs)} will be initialized.')
    else:
        resp = input(f'{len(new_configs)} will be initialized. Continue? y/n')
        if resp == 'y':
            for config, dirpath in new_configs:
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                config_filepath = os.path.join(dirpath, 'config.yaml')
                if os.path.exists(config_filepath) and not args.force:
                    print(f'{config_filepath} already exists')
                    continue
                bmm.dump_yaml(config, config_filepath)
                print(f'Initialized {config_filepath}')
        else:
            print('No dataset was initialized.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='initialize train datasets')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    args = parser.parse_args()
    main()
