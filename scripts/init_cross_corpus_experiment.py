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
        seed=42,
        force=True,
    )


def init_model(arch, train_path):
    model_init.init_from_kwargs(
        arch=arch,
        train_path=arg_type_path(train_path),
        force=True,
    )


for dim, vals in dict_.items():
    p = init_train_dset(**{dim: set(vals)})
    init_model('convtasnet', p)
    for val in vals:
        p = init_train_dset(**{dim: {val}})
        init_model('convtasnet', p)
        p = init_train_dset(**{dim: {v for v in vals if v != val}})
        init_model('convtasnet', p)
        init_test_dset(**{dim: {val}})
        init_test_dset(**{dim: {v for v in vals if v != val}})
