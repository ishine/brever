import argparse
import os

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    dict_ = {
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
            noise_num=[0, 0],
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
            noise_num=[0, 0],
        )

    def init_model(arch, train_path):
        return model_init.init_from_kwargs(
            arch=arch,
            train_path=arg_type_path(train_path),
            force=args.force,
        )

    archs = ['convtasnet']
    dsets = []
    models = []
    evaluations = {}

    def add_evals(model, paths):
        if model not in evaluations.keys():
            evaluations[model] = []
        for p in paths:
            if p not in evaluations[model]:
                evaluations[model].append(p)

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

    eval_script = 'cross_corpus_eval.sh'
    with open(eval_script, 'w') as f:
        for model, test_paths in evaluations.items():
            f.write(f"bash jobs/test_model.sh {model} {' '.join(test_paths)}")
            f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('initialize cross-corpus experiment')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    main()
