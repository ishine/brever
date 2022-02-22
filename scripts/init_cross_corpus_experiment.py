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
        )


    def init_model(arch, train_path):
        return model_init.init_from_kwargs(
            arch=arch,
            train_path=arg_type_path(train_path),
        )


    archs = ['convtasnet', 'dnn']
    evaluations = []
    for dim, vals in dict_.items():
        for val in vals:
            p1 = init_train_dset(**{dim: {val}})
            p2 = init_train_dset(**{dim: {v for v in vals if v != val}})
            p3 = init_test_dset(**{dim: {val}})
            p4 = init_test_dset(**{dim: {v for v in vals if v != val}})
            for arch in archs:
                m1 = init_model(arch, p1)
                m2 = init_model(arch, p2)
                evaluations.append(f'bash jobs/test_model.sh {m1} {p3}\n')
                evaluations.append(f'bash jobs/test_model.sh {m2} {p3}\n')
                evaluations.append(f'bash jobs/test_model.sh {m1} {p4}\n')
                evaluations.append(f'bash jobs/test_model.sh {m2} {p4}\n')

    eval_script = 'cross_corpus_eval.sh'
    with open(eval_script, 'w') as f:
        f.writelines(set(evaluations))


if __name__ == '__main__':
    main()
