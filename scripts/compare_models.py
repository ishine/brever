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
        'batchnorm': ['MODEL', 'BATCHNORM', 'ON'],
        'dropout': ['MODEL', 'DROPOUT', 'ON'],
        'features': ['POST', 'FEATURES'],
        'batchsize': ['MODEL', 'BATCHSIZE'],
    }
    base_params = {
        'layers': 1,
        'stacks': 0,
        'batchnorm': False,
        'dropout': False,
        'features': ['mfcc'],
        'batchsize': 32,
    }
    for key in base_params.keys():
        if kwargs[key] is not None:
            base_params[key] = kwargs[key]
    train_path_keys = ['POST', 'PATH', 'TRAIN']
    val_path_keys = ['POST', 'PATH', 'VAL']
    test_path_keys = ['POST', 'PATH', 'TEST']
    if kwargs['centered'] + kwargs['onlyreverb'] + kwargs['noltas'] > 1:
        raise ValueError(('Only one of centered, onlyreverb and noltas is '
                          'allowed at a time'))
    elif kwargs['centered']:
        train_path = 'data\\processed\\centered_training'
        val_path = 'data\\processed\\centered_validation'
        test_path = 'data\\processed\\centered_testing'
    elif kwargs['onlyreverb']:
        train_path = 'data\\processed\\onlyreverb_training'
        val_path = 'data\\processed\\onlyreverb_validation'
        test_path = 'data\\processed\\onlyreverb_testing'
    elif kwargs['noltas']:
        train_path = 'data\\processed\\noltas_training'
        val_path = 'data\\processed\\noltas_validation'
        test_path = 'data\\processed\\noltas_testing'
    else:
        train_path = 'data\\processed\\training'
        val_path = 'data\\processed\\validation'
        test_path = 'data\\processed\\testing'
    values = []
    models_sorted = []
    for model_id in os.listdir('models'):
        pesq_filepath = os.path.join('models', model_id, 'eval_PESQ.npy')
        mse_filepath = os.path.join('models', model_id, 'eval_MSE.npy')
        if (not os.path.exists(pesq_filepath)
                or not os.path.exists(mse_filepath)):
            continue
        config_file = os.path.join('models', model_id, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if (get_dict_field(config, train_path_keys) != train_path
                or get_dict_field(config, val_path_keys) != val_path
                or get_dict_field(config, test_path_keys) != test_path):
            continue
        invalid = False
        for filter_dim, filter_val in base_params.items():
            if (get_dict_field(config, key_dict[filter_dim]) != filter_val
                    and filter_dim != kwargs['dimension']):
                invalid = True
                break
        if invalid:
            continue
        val = get_dict_field(config, key_dict[kwargs['dimension']])
        if val not in values:
            values.append(val)
            models_sorted.append([model_id])
        else:
            index = values.index(val)
            models_sorted[index].append(model_id)

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
    if kwargs['dimension'] == 'features':
        i_values_sorted = np.argsort(np.mean(pesqs, axis=(1, 2)))
    else:
        i_values_sorted = np.argsort(values)

    print(f'Comparing {len(models_sorted)} group(s) of models')
    for i, j in enumerate(i_values_sorted):
        print((f'Group {i+1} associated with dimension value {values[j]} '
               f'contains {len(models_sorted[j])} model(s):'))
        for model in models_sorted[j]:
            print(f'  {model}')

    for metrics, ylabel, filetag in zip(
                [mses, pesqs],
                ['MSE', '$\\Delta PESQ$'],
                ['mse', 'pesq'],
            ):
        for axis, (xticklabels, xlabel, filetag_) in enumerate(zip(
                    [room_names, snrs],
                    ['room', 'SNR (dB)'],
                    ['rooms', 'snrs'],
                )):
            if ((kwargs['onlyreverb'] or kwargs['noltas'])
                    and xticklabels == snrs):
                continue
            fig, ax = plt.subplots()
            for i, j in enumerate(i_values_sorted):
                metric = metrics[j]
                val = values[j]
                data = metric.mean(axis=axis)
                data = np.hstack((data, data.mean()))
                label = f"{kwargs['dimension']} = {val}"
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
            fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare trained models.')
    parser.add_argument('dimension',
                        help=('Parameter dimension of models to compare.'))
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
    parser.add_argument('--centered', action='store_true',
                        help=('Centered target.'))
    parser.add_argument('--onlyreverb', action='store_true',
                        help=('Only reverb.'))
    parser.add_argument('--noltas', action='store_true',
                        help=('No LTAS.'))
    args = parser.parse_args()

    main(
        dimension=args.dimension,
        layers=args.layers,
        stacks=args.stacks,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        features=args.features,
        batchsize=args.batchsize,
        centered=args.centered,
        onlyreverb=args.onlyreverb,
        noltas=args.noltas,
    )
