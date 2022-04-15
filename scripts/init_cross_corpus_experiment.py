import argparse
import os

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
            duration=36000,
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
        kwargs = {}
        if arch == 'convtasnet-k=2':
            arch = 'convtasnet'
            kwargs['sources'] = ['foreground', 'background']
        elif arch == 'convtasnet-big':
            arch = 'convtasnet'
            kwargs['filters'] = 512
            kwargs['hidden_channels'] = 512
            kwargs['layers'] = 8
            kwargs['repeats'] = 3
        return model_init.init_from_kwargs(
            arch=arch,
            train_path=arg_type_path(train_path),
            force=args.force,
            **kwargs,
        )

    # archs = ['dnn', 'convtasnet', 'convtasnet-k=2', 'convtasnet-big']
    archs = ['dnn', 'convtasnet', 'convtasnet-k=2']
    dsets = []
    models = []
    evaluations = []

    def add_evals(model, paths):
        for p in paths:
            evaluations.append(f'bash jobs/test_model.sh {model} {p}\n')

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

    eval_script = 'cross_corpus_eval.sh'
    with open(eval_script, 'w') as f:
        f.writelines(set(evaluations))

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
