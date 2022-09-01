import os
import subprocess

from brever.args import ModelArgParser
from brever.config import ModelFinder, get_config


def main():
    finder = ModelFinder()
    matching_models, _ = finder.find_from_args(args)

    models = []
    for model in matching_models:
        loss_file = os.path.join(model, 'losses.npz')

        if os.path.exists(loss_file):
            continue

        config_path = os.path.join(model, 'config.yaml')
        config = get_config(config_path)
        train_path = config.TRAINING.PATH
        mix_info_file = os.path.join(train_path, 'mixture_info.json')

        if os.path.exists(mix_info_file):
            models.append(model)

    if args.pipe:
        print(' '.join(models), end='')
    else:
        for model in models:
            print(model)

    if models and args.evaluate:
        print(f'{len(models)} models will be trained.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for model in models:
                subprocess.call([
                    'bash',
                    'jobs/train_model.sh',
                    model,
                ])
        else:
            print('No model was trained')


if __name__ == '__main__':
    parser = ModelArgParser(req=False, description='find models')
    parser.add_argument('--pipe', action='store_true',
                        help='output as one line to pipe to another command')
    parser.add_argument('--train', action='store_true',
                        help='launch trainings')
    args = parser.parse_args()
    main()
