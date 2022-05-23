import argparse
import os
import itertools
import random

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    dict_ = {
        'speakers': [
            'timit_.*',
            'libri_.*',
            'wsj0_.*',
            'clarity_.*',
            'vctk_.*',
        ],
        'noises': [
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

    def init_train_dset(
        speakers={'timit_.*'},
        noises={'dcase_.*'},
        rooms={'surrey_.*'},
    ):
        return dset_init.init_from_kwargs(
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
        speakers={'timit_.*'},
        noises={'dcase_.*'},
        rooms={'surrey_.*'},
    ):
        return dset_init.init_from_kwargs(
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

    def init_model(arch, train_path):
        return model_init.init_from_kwargs(
            arch=arch,
            train_path=arg_type_path(train_path),
            force=args.force,
        )

    archs = ['dnn', 'convtasnet']
    dsets = []
    models = []
    evaluations = {}

    def add_evals(model, paths):
        if model not in evaluations.keys():
            evaluations[model] = []
        for p in paths:
            if p not in evaluations[model]:
                evaluations[model].append(p)

    # single mismatch
    for dim, vals in dict_.items():
        # test paths
        test_paths = []
        for val in vals:
            p = init_test_dset(**{dim: {val}})
            dsets.append(p)
            test_paths.append(p)
        # train paths
        for val in vals:
            p1 = init_train_dset(**{dim: {val}})
            p2 = init_train_dset(**{dim: {v for v in vals if v != val}})
            dsets.append(p1)
            dsets.append(p2)
            # models
            for arch in archs:
                m1 = init_model(arch, p1)
                m2 = init_model(arch, p2)
                models.append(m1)
                models.append(m2)
                # evaluations
                add_evals(m1, test_paths)
                add_evals(m2, test_paths)
        # models for alternative definition of generalization gap
        p3 = init_train_dset(**{dim: set(vals)})
        dsets.append(p3)
        for arch in archs:
            m3 = init_model(arch, p3)
            models.append(m3)
            add_evals(m3, test_paths)

    # double mismatch
    random.seed(0)
    for dim in dict_.keys():
        random.shuffle(dict_[dim])
    for dims in itertools.combinations(dict_.keys(), 2):
        # test paths
        test_paths = []
        for vals in zip(dict_[dims[0]], dict_[dims[1]]):
            kwargs = {dim: {val} for dim, val in zip(dims, vals)}
            p = init_test_dset(**kwargs)
            dsets.append(p)
            test_paths.append(p)
        # train paths
        for vals in zip(dict_[dims[0]], dict_[dims[1]]):
            kwargs = {dim: {val} for dim, val in zip(dims, vals)}
            p1 = init_train_dset(**kwargs)
            kwargs = {
                dims[0]: {v for v in dict_[dims[0]] if v != vals[0]},
                dims[1]: {v for v in dict_[dims[1]] if v != vals[1]},
            }
            p2 = init_train_dset(**kwargs)
            dsets.append(p1)
            dsets.append(p2)
            # models
            for arch in archs:
                m1 = init_model(arch, p1)
                m2 = init_model(arch, p2)
                models.append(m1)
                models.append(m2)
                # evaluations
                add_evals(m1, test_paths)
                add_evals(m2, test_paths)
        # models for alternative definition of generalization gap
        kwargs = {
            dims[0]: set(dict_[dims[0]]),
            dims[1]: set(dict_[dims[1]]),
        }
        p3 = init_train_dset(**kwargs)
        dsets.append(p3)
        for arch in archs:
            m3 = init_model(arch, p3)
            models.append(m3)
            add_evals(m3, test_paths)

    # triple mismatch
    random.seed(42)
    for dim in dict_.keys():
        random.shuffle(dict_[dim])
    # test paths
    test_paths = []
    for vals in zip(*dict_.values()):
        kwargs = {dim: {val} for dim, val in zip(dict_.keys(), vals)}
        p = init_test_dset(**kwargs)
        dsets.append(p)
        test_paths.append(p)
    # train paths
    for vals in zip(*dict_.values()):
        kwargs = {dim: {val} for dim, val in zip(dict_.keys(), vals)}
        p1 = init_train_dset(**kwargs)
        kwargs = {
            dim: {v for v in dict_[dim] if v != val}
            for dim, val in zip(dict_.keys(), vals)
        }
        p2 = init_train_dset(**kwargs)
        dsets.append(p1)
        dsets.append(p2)
        # models
        for arch in archs:
            m1 = init_model(arch, p1)
            m2 = init_model(arch, p2)
            models.append(m1)
            models.append(m2)
            # evaluations
            add_evals(m1, test_paths)
            add_evals(m2, test_paths)
    # models for alternative definition of generalization gap
    kwargs = {
        dim: set(vals)
        for dim, vals in dict_.items()
    }
    print(kwargs)
    p3 = init_train_dset(**kwargs)
    dsets.append(p3)
    for arch in archs:
        m3 = init_model(arch, p3)
        models.append(m3)
        add_evals(m3, test_paths)

    # finally add a model trained on everything and tested on everything
    test_paths = []
    for dim, vals in dict_.items():  # first one test dataset per database
        for val in vals:
            p = init_test_dset(**{dim: {val}})
            dsets.append(p)
            test_paths.append(p)
    # then one test dataset mixing everything
    p = init_test_dset(**{dim: set(vals) for dim, vals in dict_.items()})
    dsets.append(p)
    test_paths.append(p)
    # now one train dataset mixing everything
    p0 = init_train_dset(**{dim: set(vals) for dim, vals in dict_.items()})
    dsets.append(p0)
    # finally models
    for arch in archs:
        m0 = init_model(arch, p0)
        models.append(m0)
        # evaluation
        add_evals(m0, test_paths)

    eval_script = 'cross_corpus_eval.sh'
    with open(eval_script, 'w') as f:
        for model, test_paths in evaluations.items():
            f.write(f"bash jobs/test_model.sh {model} {' '.join(test_paths)}")
            f.write("\n")

    for model_id in os.listdir(model_init.dir_):
        model_path = os.path.join(model_init.dir_, model_id)
        if model_path not in models:
            print('the following model was found in the system and was '
                  f'not attempted to be initialized: {model_path}')
    for kind in ['test', 'train']:
        subdir = os.path.join(dset_init.dir_, kind)
        for dset_id in os.listdir(subdir):
            dset_path = os.path.join(subdir, dset_id).replace('\\', '/')
            if dset_path not in dsets:
                print('the following dataset was found in the system and was '
                      f'not attempted to be initialized: {dset_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('initialize cross-corpus experiment')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    main()
