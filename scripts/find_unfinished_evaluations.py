import os
import itertools
import subprocess
import h5py

from brever.args import ModelArgParser
from brever.config import DatasetInitializer, ModelFinder


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


def get_test_dset(
    dset_initializer,
    speakers={'timit_.*'},
    noises={'dcase_.*'},
    rooms={'surrey_.*'},
):
    return dset_initializer.get_path_from_kwargs(
        kind='test',
        speakers=speakers,
        noises=noises,
        rooms=rooms,
        speech_files=[0.8, 1.0],
        noise_files=[0.8, 1.0],
        room_files='odd',
        duration=3600,
        seed=42,
    )


def get_all_test_dsets(dset_initializer):
    test_paths = []
    for i, j, k in itertools.product(range(5), repeat=3):
        test_path = get_test_dset(
            dset_initializer,
            speakers={databases[0]['databases'][i]},
            noises={databases[1]['databases'][j]},
            rooms={databases[2]['databases'][k]},
        )
        test_paths.append(test_path)
    return test_paths


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    all_models, _ = ModelFinder().find()

    models = []
    for model in all_models:
        loss_file = os.path.join(model, 'losses.npz')
        score_file = os.path.join(model, 'scores.hdf5')

        if not os.path.exists(loss_file):
            continue

        if not os.path.exists(score_file):
            models.append(model)
        else:
            with h5py.File(score_file, 'r') as f:
                n_evals = len(f['data/datasets/test'].keys())
            if n_evals == 125:
                continue
            else:
                models.append(model)

    if args.pipe:
        print(' '.join(models), end='')
    else:
        for model in models:
            print(model)

    if models and args.evaluate:
        test_paths = get_all_test_dsets(dset_init)

        print(f'{len(models)} models will be evaluated.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for model in models:
                subprocess.call([
                    'bash',
                    'jobs/test_model.sh',
                    model,
                    *test_paths
                ])
        else:
            print('No dataset was evaluated')


if __name__ == '__main__':
    parser = ModelArgParser(req=False, description='find models')
    parser.add_argument('--pipe', action='store_true',
                        help='output as one line to pipe to another command')
    parser.add_argument('--evaluate', action='store_true',
                        help='launch evaluations')
    args = parser.parse_args()
    main()
