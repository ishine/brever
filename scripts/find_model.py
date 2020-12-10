import os
import shutil

from brever.modelmanagement import find_model, ModelFilterArgParser


def main(delete=False, **kwargs):
    models = find_model(**kwargs)

    trained = []
    untrained = []

    for model_id in models:
        train_loss = os.path.join('models', model_id, 'train_losses.npy')
        val_loss = os.path.join('models', model_id, 'val_losses.npy')
        if os.path.exists(train_loss) and os.path.exists(val_loss):
            trained.append(model_id)
        else:
            untrained.append(model_id)

    print(f'{len(models)} total models found')
    print(f'{len(trained)} trained models:')
    for model_id in trained:
        print(model_id)
    print(f'{len(untrained)} untrained models:')
    for model_id in untrained:
        print(model_id)

    if models and delete:
        print(f'{len(models)} will be deleted.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for model_id in models:
                model_dir = os.path.join('models', model_id)
                shutil.rmtree(model_dir)
                print(f'Deleted {model_dir}')
        else:
            print('No model was deleted')


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='find models')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='delete found models')
    filter_args, extra_args = parser.parse_args()
    main(delete=extra_args.delete, **vars(filter_args))
