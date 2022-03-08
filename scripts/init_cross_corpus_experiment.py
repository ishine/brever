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
            'ieee',
            'arctic',
            'vctk',
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
            duration=1800,
            seed=42,
            force=args.force,
        )

    def init_model(arch, train_path):
        return model_init.init_from_kwargs(
            arch=arch,
            train_path=arg_type_path(train_path),
            force=args.force,
        )

    archs = ['convtasnet', 'dnn']
    dsets = []
    models = []
    evaluations = []
    for dim, vals in dict_.items():
        for val in vals:
            p1 = init_train_dset(**{dim: {val}})
            p2 = init_train_dset(**{dim: {v for v in vals if v != val}})
            p3 = init_test_dset(**{dim: {val}})
            p4 = init_test_dset(**{dim: {v for v in vals if v != val}})
            dsets.append(p1)
            dsets.append(p2)
            dsets.append(p3)
            dsets.append(p4)
            for arch in archs:
                m1 = init_model(arch, p1)
                m2 = init_model(arch, p2)
                models.append(m1)
                models.append(m2)
                evaluations.append(f'bash jobs/test_model.sh {m1} {p3}\n')
                evaluations.append(f'bash jobs/test_model.sh {m2} {p3}\n')
                evaluations.append(f'bash jobs/test_model.sh {m1} {p4}\n')
                evaluations.append(f'bash jobs/test_model.sh {m2} {p4}\n')

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
