import argparse
import itertools

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer


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

    p_test = dset_init.init_from_kwargs(
        kind='test',
        speakers={'libri_.*'},
        noises={'dcase_.*'},
        rooms={'surrey_.*'},
        speech_files=[0.8, 1.0],
        noise_files=[0.8, 1.0],
        room_files='odd',
        duration=3600,
        seed=42,
        force=args.force,
    )

    hyperparams = {
        'batch_size': [1, 8, 16],
        'segment_length': [0.0, 1.0, 4.0],
    }

    evaluations = []
    for values in itertools.product(*hyperparams.values()):
        kwargs = dict(zip(hyperparams.keys(), values))
        m = model_init.init_from_kwargs(
            arch='convtasnet',
            train_path=arg_type_path(p_train),
            force=args.force,
            **kwargs,
        )
        evaluations.append(f'bash jobs/test_model.sh {m} {p_test}\n')

    eval_script = 'conv_tasnet_eval.sh'
    with open(eval_script, 'w') as f:
        f.writelines(set(evaluations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('initialize conv-tasnet models')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    main()
