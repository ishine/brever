import os
import argparse

import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find models.')
    parser.add_argument('--layers', type=int,
                        help=('Fixed number of layers.'))
    parser.add_argument('--stacks', type=int,
                        help=('Fixed number of stacks.'))
    parser.add_argument('--batchnorm', type=int,
                        help=('Fixed batchnorm.'))
    parser.add_argument('--dropout', type=int,
                        help=('Fixed dropout.'))
    parser.add_argument('--batchsize', type=int,
                        help=('Fixed batchsize.'))
    parser.add_argument('--features', nargs='+',
                        help=('Fixed batchsize.'))
    parser.add_argument('--centered', type=int,
                        help=('Centered target.'))
    parser.add_argument('--onlyreverb', type=int,
                        help=('Only reverb.'))
    parser.add_argument('--noltas', type=int,
                        help=('No LTAS.'))
    args = parser.parse_args()

    trained = []
    untrained = []
    for model_id in os.listdir('models'):
        config_file = os.path.join('models', model_id, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if args.layers is not None:
            if config['MODEL']['NLAYERS'] != args.layers:
                continue
        if args.stacks is not None:
            if config['POST']['STACK'] != args.stacks:
                continue
        if args.batchnorm is not None:
            if config['MODEL']['BATCHNORM']['ON'] != args.batchnorm:
                continue
        if args.dropout is not None:
            if config['MODEL']['DROPOUT']['ON'] != args.dropout:
                continue
        if args.batchsize is not None:
            if config['MODEL']['BATCHSIZE'] != args.batchsize:
                continue
        if args.features is not None:
            if set(config['POST']['FEATURES']) != set(args.features):
                continue
        if args.centered is not None:
            train_path = 'data\\processed\\centered_training'
            val_path = 'data\\processed\\centered_validation'
            train_path_ = config['POST']['PATH']['TRAIN']
            val_path_ = config['POST']['PATH']['VAL']
            if train_path != train_path_ or val_path != val_path_:
                if args.centered:
                    continue
            else:
                if not args.centered:
                    continue
        if args.onlyreverb is not None:
            train_path = 'data\\processed\\onlyreverb_training'
            val_path = 'data\\processed\\onlyreverb_validation'
            train_path_ = config['POST']['PATH']['TRAIN']
            val_path_ = config['POST']['PATH']['VAL']
            if train_path != train_path_ or val_path != val_path_:
                if args.onlyreverb:
                    continue
            else:
                if not args.onlyreverb:
                    continue
        if args.noltas is not None:
            train_path = 'data\\processed\\noltas_training'
            val_path = 'data\\processed\\noltas_validation'
            train_path_ = config['POST']['PATH']['TRAIN']
            val_path_ = config['POST']['PATH']['VAL']
            if train_path != train_path_ or val_path != val_path_:
                if args.noltas:
                    continue
            else:
                if not args.noltas:
                    continue
        train_loss = os.path.join('models', model_id, 'train_losses.npy')
        val_loss = os.path.join('models', model_id, 'val_losses.npy')
        if os.path.exists(train_loss) and os.path.exists(val_loss):
            trained.append(model_id)
        else:
            untrained.append(model_id)

    print(f'{len(trained) + len(untrained)} total models found.')
    print(f'{len(trained)} trained models:')
    for model_id in trained:
        print(model_id)
    print(f'{len(untrained)} untrained models:')
    for model_id in untrained:
        print(model_id)
