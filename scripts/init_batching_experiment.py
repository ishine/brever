import argparse
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

eval_script = 'batching_eval.sh'


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


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    p_train = dset_init.init_from_kwargs(
        kind='train',
        speakers={'libri_.*'},
        noises={'dcase_.*'},
        rooms={'surrey_.*'},
        speech_files=[0.0, 0.8],
        noise_files=[0.0, 0.8],
        room_files='even',
        duration=36000,
        seed=0,
        force=args.force,
    )

    test_paths = init_all_test_dsets(dset_init)

    seeds = [0, 1, 2, 3, 4]
    fixed_sizes = [1, 2, 4, 8, 16, 32]
    dynamic_sizes = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    batch_samplers = ['random', 'sorted', 'bucket']

    models = []

    for batch_sampler, dynamic, seed in itertools.product(
        batch_samplers,
        [False, True],
        seeds,
    ):

        if dynamic:
            batch_sizes = dynamic_sizes
        else:
            batch_sizes = fixed_sizes

        for batch_size in batch_sizes:
            m = model_init.init_from_kwargs(
                arch='convtasnet',
                train_path=arg_type_path(p_train),
                force=args.force,
                batch_size=float(batch_size),
                batch_sampler=batch_sampler,
                dynamic=dynamic,
                seed=seed,
            )
            models.append(m)

    write_eval_script(models, test_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('initialize conv-tasnet models')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    main()
