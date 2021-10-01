import os
import shutil

from brever.config import defaults
import brever.modelmanagement as bmm


def main(args, **kwargs):
    if args.trained and args.untrained:
        raise ValueError("can't use both --trained and --untrained")
    if args.tested and args.untested:
        raise ValueError("can't use both --tested and --untested")

    # first filtering of models
    pre_models = bmm.find_model(**kwargs)

    # second filtering of models based on extra argumgents
    models = []

    for model in pre_models:
        loss_file = os.path.join(model, 'losses.npz')
        score_file = os.path.join(model, 'scores.json')

        if args.untrained and os.path.exists(loss_file):
            continue
        if args.trained and not os.path.exists(loss_file):
            continue
        if args.untested and os.path.exists(score_file):
            continue
        if args.tested and not os.path.exists(score_file):
            continue

        models.append(model)

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
    parser = bmm.ModelFilterArgParser(description='find models')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='delete found models')
    parser.add_argument('--trained', action='store_true',
                        help='only show trained models')
    parser.add_argument('--untrained', action='store_true',
                        help='only show untrained models')
    parser.add_argument('--tested', action='store_true',
                        help='only show tested models')
    parser.add_argument('--untested', action='store_true',
                        help='only show untested models')
    filter_args, extra_args = parser.parse_args()
    main(extra_args, **vars(filter_args))
