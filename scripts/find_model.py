import os

from brever.modelmanagement import find_model, ModelFilterArgParser


def main(**kwargs):
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


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='find models')
    filter_args, _ = parser.parse_args()
    main(**vars(filter_args))
