import os
import argparse
import json

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
    parser.add_argument('--traincentered', type=int,
                        help=('Trained on centered target.'))
    parser.add_argument('--testcentered', type=int,
                        help=('Tested on centered target.'))
    parser.add_argument('--testonlyreverb', type=int,
                        help=('Tested on only reverberation.'))
    parser.add_argument('--testonlydiffuse', type=int,
                        help=('Tested on only diffuse nosie.'))
    parser.add_argument('--trainbig', type=int,
                        help=('Trained on big dataset.'))
    parser.add_argument('--testbig', type=int,
                        help=('Tested on big dataset.'))
    parser.add_argument('--showconfig', action='store_true',
                        help=('Print configs.'))
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
        if args.traincentered is not None:
            if args.traincentered:
                if 'centered' not in config['POST']['PATH']['TRAIN']:
                    continue
            else:
                if 'centered' in config['POST']['PATH']['TRAIN']:
                    continue
        if args.testcentered is not None:
            if args.testcentered:
                if 'centered' not in config['POST']['PATH']['TEST']:
                    continue
            else:
                if 'centered' in config['POST']['PATH']['TEST']:
                    continue
        if args.testonlyreverb is not None:
            if args.testonlyreverb:
                if 'onlyreverb' not in config['POST']['PATH']['TEST']:
                    continue
            else:
                if 'onlyreverb' in config['POST']['PATH']['TEST']:
                    continue
        if args.testonlydiffuse is not None:
            if args.testonlydiffuse:
                if 'onlydiffuse' not in config['POST']['PATH']['TEST']:
                    continue
            else:
                if 'onlydiffuse' in config['POST']['PATH']['TEST']:
                    continue
        if args.testbig is not None:
            if args.testbig:
                if not config['POST']['PATH']['TEST'].endswith('big'):
                    continue
            else:
                if config['POST']['PATH']['TEST'].endswith('big'):
                    continue
        if args.trainbig is not None:
            if args.trainbig:
                if not config['POST']['PATH']['TRAIN'].endswith('big'):
                    continue
            else:
                if config['POST']['PATH']['TRAIN'].endswith('big'):
                    continue
        train_loss = os.path.join('models', model_id, 'train_losses.npy')
        val_loss = os.path.join('models', model_id, 'val_losses.npy')
        if os.path.exists(train_loss) and os.path.exists(val_loss):
            trained.append(model_id)
        else:
            untrained.append(model_id)
        if args.showconfig:
            print(json.dumps(config, indent=4))

    print(f'{len(trained) + len(untrained)} total models found.')
    print(f'{len(trained)} trained models:')
    for model_id in trained:
        print(model_id)
    print(f'{len(untrained)} untrained models:')
    for model_id in untrained:
        print(model_id)
