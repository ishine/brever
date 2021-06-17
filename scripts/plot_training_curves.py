import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from brever.modelmanagement import ModelFilterArgParser, find_model


def smooth(data, sigma=50):
    df = np.subtract.outer(np.arange(len(data)), np.arange(len(data)))
    filtering_mat = np.exp(-0.5*(df/sigma)**2)/(sigma*(2*np.pi)**0.5)
    filtering_mat = np.tril(filtering_mat)
    filtering_mat /= filtering_mat.sum(axis=1, keepdims=True)
    return filtering_mat@data


def main(models, **kwargs):
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')

    possible_models = find_model(**kwargs)
    models = [model for model in models if model in possible_models]
    for model in models:
        print(model)

    for smooth_func in [lambda x: x, smooth]:
        plt.figure(figsize=(16, 8))
        for i, model in enumerate(models):
            path = os.path.join(model, 'losses.npz')
            data = np.load(path)
            label = f'{os.path.basename(model)[:3]}...'
            l, = plt.plot(smooth_func(data['train'], label=label))
            _, = plt.plot(smooth_func(data['val'], '--', color=l.get_color()))
            plt.legend(ncol=10)
            plt.grid()

    plt.figure(figsize=(16, 8))
    for i, model in enumerate(models):
        path = os.path.join(model, 'losses.npz')
        data = smooth(np.load(path)['train'])
        slope = np.zeros(len(data))
        strip = 100
        for i in range(len(data)-strip):
            x = np.arange(strip)
            y = data[i:i+strip]
            slope[i+strip] = (np.sum((x - x.mean())*(y - y.mean())) /
                              np.sum((x - x.mean())*(x - x.mean())))
        plt.plot(np.log10(abs(slope)))
    plt.grid()

    plt.show()


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='plot training curves')
    parser.add_argument('input', nargs='+',
                        help='list of models whose curves to plot')
    filter_args, args = parser.parse_args()

    model_dirs = []
    for input_ in args.input:
        if not glob(input_):
            print(f'Model not found: {input_}')
        model_dirs += glob(input_)
    main(model_dirs, **vars(filter_args))
