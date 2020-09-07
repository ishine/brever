import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml

from brever.modelmanagement import get_dict_field


def main(**kwargs):
    key_dict = {
        'layers': ['MODEL', 'NLAYERS'],
        'stacks': ['POST', 'STACK'],
        'features': ['POST', 'FEATURES'],
        'batchsize': ['MODEL', 'BATCHSIZE'],
    }
    base_params = {
        'layers': 1,
        'stacks': 4,
        'features': ['mfcc'],
        'batchsize': 32,
    }
    values = [
        {
            'batchnorm': False,
            'dropout': False,
        },
        {
            'batchnorm': False,
            'dropout': True,
        },
        {
            'batchnorm': True,
            'dropout': False,
        },
        {
            'batchnorm': True,
            'dropout': True,
        },
    ]
    for key in base_params.keys():
        if kwargs[key] is not None:
            base_params[key] = kwargs[key]
    train_path_keys = ['POST', 'PATH', 'TRAIN']
    val_path_keys = ['POST', 'PATH', 'VAL']
    test_path_keys = ['POST', 'PATH', 'TEST']
    if kwargs['centered'] + kwargs['onlyreverb'] > 1:
        raise ValueError(('Only one of centered and onlyreverb is allowed at a'
                          'time'))
    elif kwargs['centered']:
        train_path = 'data\\processed\\centered_training'
        val_path = 'data\\processed\\centered_validation'
        test_path = 'data\\processed\\centered_testing'
    elif kwargs['onlyreverb']:
        train_path = 'data\\processed\\onlyreverb_training'
        val_path = 'data\\processed\\onlyreverb_validation'
        test_path = 'data\\processed\\onlyreverb_testing'
    else:
        train_path = 'data\\processed\\training'
        val_path = 'data\\processed\\validation'
        test_path = 'data\\processed\\testing'
    if kwargs['testbig']:
        test_path = test_path + '_big'
    if kwargs['trainbig']:
        train_path = train_path + '_big'
        val_path = val_path + '_big'
    models_sorted = [[] for i in range(len(values))]
    for model_id in os.listdir('models'):
        pesq_filepath = os.path.join('models', model_id, 'eval_PESQ.npy')
        mse_filepath = os.path.join('models', model_id, 'eval_MSE.npy')
        config_file = os.path.join('models', model_id, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if (get_dict_field(config, train_path_keys) != train_path
                or get_dict_field(config, val_path_keys) != val_path
                or get_dict_field(config, test_path_keys) != test_path):
            continue
        invalid = False
        for filter_dim, filter_val in base_params.items():
            if get_dict_field(config, key_dict[filter_dim]) != filter_val:
                invalid = True
                break
        if invalid:
            continue
        invalid = True
        for i, val in enumerate(values):
            if (val['batchnorm'] == config['MODEL']['BATCHNORM']['ON']
                    and val['dropout'] == config['MODEL']['DROPOUT']['ON']):
                invalid = False
                break
        if invalid:
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
            for i, (metric, val) in enumerate(zip(metrics, values)):
                data = metric.mean(axis=axis)
                data = np.hstack((data, data.mean()))
                label = f'{val}'
                x = np.arange(len(data)) + (i - (n-1)/2)*width
                x[-1] = x[-1] + 2*width
                ax.bar(
                    x=x,
                    height=data,
                    width=width,
                    label=label,
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
    parser = argparse.ArgumentParser(description='Compare regularization.')
    parser.add_argument('--layers', type=int,
                        help=('Fixed number of layers.'))
    parser.add_argument('--stacks', type=int,
                        help=('Fixed number of stacks.'))
    parser.add_argument('--features', nargs='+',
                        help=('Fixed feature set.'))
    parser.add_argument('--batchsize', type=int,
                        help=('Fixed batchsize.'))
    parser.add_argument('--centered', action='store_true',
                        help=('Centered target.'))
    parser.add_argument('--onlyreverb', action='store_true',
                        help=('Only reverb.'))
    parser.add_argument('--testbig', action='store_true',
                        help=('Use models evaluated on big test datasets.'))
    parser.add_argument('--trainbig', action='store_true',
                        help=('Use models trained on big train datasets.'))
    args = parser.parse_args()

    main(
        layers=args.layers,
        stacks=args.stacks,
        features=args.features,
        batchsize=args.batchsize,
        centered=args.centered,
        onlyreverb=args.onlyreverb,
        testbig=args.testbig,
        trainbig=args.trainbig,
    )