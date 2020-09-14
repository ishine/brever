import os
import argparse

from brever.modelmanagement import find_model


def main(args):
    models = find_model(**vars(args))

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
    parser = argparse.ArgumentParser(description='find models')
    parser.add_argument('--layers', type=int,
                        help='number of layers')
    parser.add_argument('--stacks', type=int,
                        help='number of extra stacks')
    parser.add_argument('--batchnorm', type=lambda x: bool(int(x)),
                        help='batchnorm toggle')
    parser.add_argument('--dropout', type=lambda x: bool(int(x)),
                        help='dropout toggle')
    parser.add_argument('--batchsize', type=int,
                        help='batchsize')
    parser.add_argument('--features', type=lambda x: set(x.split(' ')),
                        help='feature set')
    parser.add_argument('--train-path',
                        help='training dataset path')
    parser.add_argument('--val-path',
                        help='validation dataset path')
    args = parser.parse_args()
    main(args)
