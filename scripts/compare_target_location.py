import os
import argparse
import colorsys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import yaml

from brever.modelmanagement import get_dict_field


def main(**kwargs):
    key_dict = {
        'layers': ['MODEL', 'NLAYERS'],
        'stacks': ['POST', 'STACK'],
        'batchnorm': ['MODEL', 'BATCHNORM', 'ON'],
        'dropout': ['MODEL', 'DROPOUT', 'ON'],
        'features': ['POST', 'FEATURES'],
        'batchsize': ['MODEL', 'BATCHSIZE'],
    }
    base_params = {
        'layers': 1,
        'stacks': 4,
        'batchnorm': False,
        'dropout': True,
        'batchsize': 32,
    }
    values = [
        {
            'features': ['ic'],
            'train_path': 'data\\processed\\training',
        },
        {
            'features': ['ic'],
            'train_path': 'data\\processed\\centered_training',
        },
        {
            'features': ['ild', 'itd', 'ic'],
            'train_path': 'data\\processed\\training',
        },
        {
            'features': ['ild', 'itd', 'ic'],
            'train_path': 'data\\processed\\centered_training',
        },
        {
            'features': ['mfcc', 'pdf'],
            'train_path': 'data\\processed\\training',
        },
        {
            'features': ['mfcc', 'pdf'],
            'train_path': 'data\\processed\\centered_training',
        },
        {
            'features': ['mfcc', 'ic'],
            'train_path': 'data\\processed\\training',
        },
        {
            'features': ['mfcc', 'ic'],
            'train_path': 'data\\processed\\centered_training',
        },
    ]

    for key in base_params.keys():
        if kwargs[key] is not None:
            base_params[key] = kwargs[key]
    train_path_keys = ['POST', 'PATH', 'TRAIN']
    test_path_keys = ['POST', 'PATH', 'TEST']
    models_sorted = [[] for i in range(len(values))]
    for model_id in os.listdir('models'):
        pesq_filepath = os.path.join('models', model_id, 'eval_PESQ.npy')
        mse_filepath = os.path.join('models', model_id, 'eval_MSE.npy')
        config_file = os.path.join('models', model_id, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        invalid = False
        for filter_dim, filter_val in base_params.items():
            if get_dict_field(config, key_dict[filter_dim]) != filter_val:
                invalid = True
                break
        if invalid:
            continue
        features = get_dict_field(config, key_dict['features'])
        train_path = get_dict_field(config, train_path_keys)
        test_path = get_dict_field(config, test_path_keys)
        invalid = True
        for i, val in enumerate(values):
            if (set(val['features']) == set(features)
                    and val['train_path'] == train_path):
                invalid = False
                break
        if invalid:
            continue
        if kwargs['testcenter']:
            if kwargs['testbig']:
                if kwargs['onlyreverb']:
                    if test_path != 'data\\processed\\onlyreverb_centered_testing_big':
                        continue
                elif kwargs['onlydiffuse']:
                    if test_path != 'data\\processed\\onlydiffuse_centered_testing_big':
                        continue
                else:
                    if test_path != 'data\\processed\\centered_testing_big':
                        continue
            else:
                if kwargs['onlyreverb']:
                    if test_path != 'data\\processed\\onlyreverb_centered_testing':
                        continue
                elif kwargs['onlydiffuse']:
                    if test_path != 'data\\processed\\onlydiffuse_centered_testing':
                        continue
                else:
                    if test_path != 'data\\processed\\centered_testing':
                        continue
        else:
            if kwargs['testbig']:
                if kwargs['onlyreverb']:
                    if test_path != 'data\\processed\\onlyreverb_testing_big':
                        continue
                elif kwargs['onlydiffuse']:
                    if test_path != 'data\\processed\\onlydiffuse_testing_big':
                        continue
                else:
                    if test_path != 'data\\processed\\testing_big':
                        continue
            else:
                if kwargs['onlyreverb']:
                    if test_path != 'data\\processed\\onlyreverb_testing':
                        continue
                elif kwargs['onlydiffuse']:
                    print(test_path)
                    if test_path != 'data\\processed\\onlydiffuse_testing':
                        continue
                else:
                    if test_path != 'data\\processed\\testing':
                        continue
        if (not os.path.exists(pesq_filepath)
                or not os.path.exists(mse_filepath)):
            print((f'Model {model_id} is attempted to be compared but is not '
                   'evaluated!'))
            continue
        models_sorted[i].append(model_id)

    pesqs = []
    mses = []
    for models in models_sorted:
        pesq = []
        mse = []
        for model_id in models:
            pesq_filepath = os.path.join('models', model_id, 'eval_PESQ.npy')
            mse_filepath = os.path.join('models', model_id, 'eval_MSE.npy')
            pesq.append(np.load(pesq_filepath))
            mse.append(np.load(mse_filepath))
        pesqs.append(np.asarray(pesq).mean(axis=0))
        mses.append(np.asarray(mse).mean(axis=0))

    snrs = [0, 3, 6, 9, 12, 15]
    room_names = [
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
    ]
    n = len(pesqs)
    width = 1/(n+1)
    new_colors = []
    for color in plt.rcParams['axes.prop_cycle'].by_key()['color']:
        for amount in [1, 1]:
            r, g, b = matplotlib.colors.to_rgb(color)
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            r, g, b = colorsys.hls_to_rgb(h, 1 - amount*(1-l), s)
            new_color = (r, g, b)
            new_colors.append(new_color)
    print(f'Comparing {len(models_sorted)} group(s) of models')
    for i, models in enumerate(models_sorted):
        print((f'Group {i+1} associated with dimension value {values[i]} '
               f'contains {len(models)} model(s):'))
        for model in models:
            print(f'  {model}')

    for metrics, ylabel, filetag in zip(
                [mses, pesqs],
                ['MSE', r'$\Delta PESQ$'],
                ['mse', 'pesq'],
            ):
        fig, axes = plt.subplots(1, 2, sharey=True)
        for axis, (ax, xticklabels, xlabel, filetag_) in enumerate(zip(
                    axes[::-1],
                    [room_names, snrs],
                    ['room', 'SNR (dB)'],
                    ['rooms', 'snrs'],
                )):
            if kwargs['onlyreverb'] and xticklabels == snrs:
                continue
            for i, (metric, val) in enumerate(zip(metrics, values)):
                data = metric.mean(axis=axis)
                data = np.hstack((data, data.mean()))
                label = (f'{val["features"]}_'
                         f'{os.path.basename(val["train_path"])}')
                x = np.arange(len(data)) + (i - (n-1)/2)*width
                x[-1] = x[-1] + 2*width
                if i % 2 == 0:
                    hatch = ''
                else:
                    hatch = '////'
                ax.bar(
                    x=x,
                    height=data,
                    width=width,
                    label=label,
                    hatch=hatch,
                    color=new_colors[i],
                )
            xticks = np.arange(len(xticklabels) + 1, dtype=float)
            xticks[-1] = xticks[-1] + 2*width
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels + ['mean'])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.yaxis.set_tick_params(labelleft=True)
        fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Compare target location '
                                                  'paradigms.'))
    parser.add_argument('--layers', type=int,
                        help=('Fixed number of layers.'))
    parser.add_argument('--stacks', type=int,
                        help=('Fixed number of stacks.'))
    parser.add_argument('--batchnorm', type=int,
                        help=('Fixed batchnorm.'))
    parser.add_argument('--dropout', type=int,
                        help=('Fixed dropout.'))
    parser.add_argument('--features', nargs='+',
                        help=('Fixed feature set.'))
    parser.add_argument('--batchsize', type=int,
                        help=('Fixed batchsize.'))
    parser.add_argument('--testcenter', action='store_true',
                        help=('Use models tested on fixed target.'))
    parser.add_argument('--testbig', action='store_true',
                        help=('Use models evaluated on big test datasets.'))
    parser.add_argument('--onlyreverb', action='store_true',
                        help=('Only reverb.'))
    parser.add_argument('--onlydiffuse', action='store_true',
                        help=('Only diffuse.'))
    args = parser.parse_args()

    main(
        layers=args.layers,
        stacks=args.stacks,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        features=args.features,
        batchsize=args.batchsize,
        testcenter=args.testcenter,
        testbig=args.testbig,
        onlyreverb=args.onlyreverb,
        onlydiffuse=args.onlydiffuse,
    )
