import os
import shutil

from brever.args import ModelArgParser
from brever.config import ModelFinder


def main():
    if args.trained and args.untrained:
        raise ValueError('cannot use both --trained and --untrained')
    if args.tested and args.untested:
        raise ValueError('cannot use both --tested and --untested')

    finder = ModelFinder()
    matching_models, _ = finder.find_from_args(args)

    models = []
    for model in matching_models:
        loss_file = os.path.join(model, 'losses.npz')
        score_file = os.path.join(model, 'scores.hdf5')

        if args.untrained and os.path.exists(loss_file):
            continue
        if args.trained and not os.path.exists(loss_file):
            continue
        if args.untested and os.path.exists(score_file):
            continue
        if args.tested and not os.path.exists(score_file):
            continue

        models.append(model)

    if args.pipe:
        print(' '.join(models), end='')
    else:
        for model in models:
            print(model)

    if models and args.delete:
        print(f'{len(models)} models will be deleted.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for model in models:
                shutil.rmtree(model)
                print(f'Deleted {model}')
        else:
            print('No model was deleted')


if __name__ == '__main__':
    parser = ModelArgParser(req=False, description='find models')
    parser.add_argument('--delete', action='store_true',
                        help='delete found models')
    parser.add_argument('--trained', action='store_true',
                        help='only show trained models')
    parser.add_argument('--untrained', action='store_true',
                        help='only show untrained models')
    parser.add_argument('--tested', action='store_true',
                        help='only show tested models')
    parser.add_argument('--untested', action='store_true',
                        help='only show untested models')
    parser.add_argument('--pipe', action='store_true',
                        help='output as one line to pipe to another command')
    args = parser.parse_args()
    main()
