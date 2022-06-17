import argparse
import os
import itertools

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer


databases = [
    {
        'kwarg': 'speakers',
        'databases': [
            'timit_.*',
            'libri_.*',
            'wsj0_.*',
            'clarity_.*',
            'vctk_.*',
        ],
    },
    {
        'kwarg': 'noises',
        'databases': [
            'dcase_.*',
            'noisex_.*',
            'icra_.*',
            'demand',
            'arte',
        ],
    },
    {
        'kwarg': 'rooms',
        'databases': [
            'surrey_.*',
            'ash_.*',
            'bras_.*',
            'catt_.*',
            'avil_.*',
        ],
    },
]

archs = ['dnn', 'convtasnet']

eval_script = 'cross_corpus_eval.sh'


def init_train_dset(
    dset_initializer,
    speakers={'timit_.*'},
    noises={'dcase_.*'},
    rooms={'surrey_.*'},
):
    return dset_initializer.init_from_kwargs(
        kind='train',
        speakers=speakers,
        noises=noises,
        rooms=rooms,
        speech_files=[0.0, 0.8],
        noise_files=[0.0, 0.8],
        room_files='even',
        duration=3*36000,
        seed=0,
        force=args.force,
    )


def init_test_dset(
    dset_initializer,
    speakers={'timit_.*'},
    noises={'dcase_.*'},
    rooms={'surrey_.*'},
):
    return dset_initializer.init_from_kwargs(
        kind='test',
        speakers=speakers,
        noises=noises,
        rooms=rooms,
        speech_files=[0.8, 1.0],
        noise_files=[0.8, 1.0],
        room_files='odd',
        duration=3600,
        seed=42,
        force=args.force,
    )


def init_model(model_initializer, arch, train_path):
    return model_initializer.init_from_kwargs(
        arch=arch,
        train_path=arg_type_path(train_path),
        force=args.force,
    )


def complement(idx_list):
    return [i for i in range(5) if i not in idx_list]


def n_eq_one(i, dims):
    return [[i], [i], [i]]


def n_eq_four(i, dims):
    return [complement([i])]*3


def build_test_index(index, dims):
    test_index = [complement(index[dim]) for dim in range(3)]
    for dim in dims:
        test_index[dim] = index[dim]
    return test_index


def build_kwargs(index):
    kwargs = {}
    for dim_dbs, dbs_idx in zip(databases, index):
        kwargs[dim_dbs['kwarg']] = {dim_dbs['databases'][i] for i in dbs_idx}
    return kwargs


def add_models(m, m_ref, models):
    for model in [m, m_ref]:
        if model not in models:
            models.append(model)


def add_train_paths(train_path, ref_train_path, train_paths):
    for p in [train_path, ref_train_path]:
        if p not in train_paths:
            train_paths.append(p)


def init_all_test_dsets(dset_initializer):
    test_paths = []
    for i, j, k in itertools.product(range(5), repeat=3):
        test_path = init_test_dset(
            dset_initializer,
            speakers={databases[0]['databases'][i]},
            noises={databases[1]['databases'][j]},
            rooms={databases[2]['databases'][k]},
        )
        test_paths.append(test_path)
    return test_paths


def write_eval_script(models, test_paths):
    with open(eval_script, 'w') as f:
        for m in models:
            f.write(f"bash jobs/test_model.sh {m} {' '.join(test_paths)}\n")


def check_deprecated_models(model_dir, models):
    for model_id in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_id)
        if model_path not in models:
            print('the following model was found in the system and was '
                  f'not attempted to be initialized: {model_path}')


def check_deprecated_dsets(dset_dir, dsets):
    for kind in ['test', 'train']:
        subdir = os.path.join(dset_dir, kind)
        for dset_id in os.listdir(subdir):
            dset_path = os.path.join(subdir, dset_id).replace('\\', '/')
            if dset_path not in dsets:
                print('the following dataset was found in the system and was '
                      f'not attempted to be initialized: {dset_path}')


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    train_paths = []
    models = []
    for index_func in [n_eq_one, n_eq_four]:
        for ndim in range(3):
            for dims in itertools.combinations(range(3), ndim):
                for i in range(5):
                    train_index = index_func(i, dims)
                    train_kwargs = build_kwargs(train_index)
                    train_path = init_train_dset(dset_init, **train_kwargs)
                    test_idx = build_test_index(train_index, dims)
                    test_kwargs = build_kwargs(test_idx)
                    ref_train_path = init_train_dset(dset_init, **test_kwargs)
                    for arch in archs:
                        m = init_model(model_init, arch, train_path)
                        m_ref = init_model(model_init, arch, ref_train_path)
                        add_models(m, m_ref, models)
                    add_train_paths(train_path, ref_train_path, train_paths)

    test_paths = init_all_test_dsets(dset_init)
    write_eval_script(models, test_paths)

    check_deprecated_models(model_init.dir_, models)
    check_deprecated_dsets(dset_init.dir_, train_paths+test_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('initialize cross-corpus experiment')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    main()
