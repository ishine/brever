import os

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer


dset_init = DatasetInitializer()
model_init = ModelInitializer()

dict_ = {
    'speakers': [
        'timit_.*',
        'libri_.*',
        'ieee',
        'arctic',
        'hint',
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
        force=True,
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
        force=True,
    )


def init_model(arch, train_path):
    return model_init.init_from_kwargs(
        arch=arch,
        train_path=arg_type_path(train_path),
        force=True,
    )


eval_script = 'cross_corpus_eval.sh'
if os.path.exists(eval_script):
    os.remove(eval_script)


def init_evaluation(model, test_path):
    with open(eval_script, 'a') as f:
        f.write(f'bash jobs/test_model.sh {model} {test_path}\n')


for arch in ['convtasnet', 'dnn']:
    for dim, vals in dict_.items():
        p0 = init_train_dset(**{dim: set(vals)})
        m0 = init_model(arch, p0)
        for val in vals:
            p1 = init_train_dset(**{dim: {val}})
            m1 = init_model(arch, p1)
            p2 = init_train_dset(**{dim: {v for v in vals if v != val}})
            m2 = init_model(arch, p2)
            p3 = init_test_dset(**{dim: {val}})
            p4 = init_test_dset(**{dim: {v for v in vals if v != val}})
            init_evaluation(m1, p3)
            init_evaluation(m2, p3)
            init_evaluation(m1, p4)
            init_evaluation(m2, p4)
