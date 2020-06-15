import os
import argparse

import yaml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
                        help=('Fixed feature set.'))
    parser.add_argument('--centered', type=int,
                        help=('Centered target.'))
    parser.add_argument('--onlyreverb', type=int,
                        help=('Only reverb.'))
    args = parser.parse_args()

    model_ids = []
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
            train = 'data\\processed\\centered_training'
            val = 'data\\processed\\centered_validation'
            test = 'data\\processed\\centered_testing'
            if (config['POST']['PATH']['TRAIN'] != train
                    or config['POST']['PATH']['VAL'] != val
                    or config['POST']['PATH']['TEST'] != test):
                if args.centered:
                    continue
            else:
                if not args.centered:
                    continue
        if args.onlyreverb is not None:
            train = 'data\\processed\\onlyreverb_training'
            val = 'data\\processed\\onlyreverb_validation'
            test = 'data\\processed\\onlyreverb_testing'
            if (config['POST']['PATH']['TRAIN'] != train
                    or config['POST']['PATH']['VAL'] != val
                    or config['POST']['PATH']['TEST'] != test):
                if args.onlyreverb:
                    continue
            else:
                if not args.onlyreverb:
                    continue
        train_loss = os.path.join('models', model_id, 'train_losses.npy')
        val_loss = os.path.join('models', model_id, 'val_losses.npy')
        if os.path.exists(train_loss) and os.path.exists(val_loss):
            model_ids.append(model_id)

    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    colors = cmap(np.linspace(0, 1, len(model_ids)))

    plt.figure(figsize=(16, 8))
    for i, model_id in enumerate(model_ids):
        path = os.path.join('models', model_id, 'train_losses.npy')
        data = np.load(path)
        last = data[-1]
        plt.plot(data, color=colors[i], label=f'{model_id[:3]}...')
        path = os.path.join('models', model_id, 'val_losses.npy')
        data = np.load(path)
        plt.plot(data, '--', color=colors[i])
        if last > data[-1]:
            print(model_id)
            config_file = os.path.join('models', model_id, 'config.yaml')
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            from pprint import pprint
            pprint(config)
    plt.legend(ncol=10)

    plt.show()
