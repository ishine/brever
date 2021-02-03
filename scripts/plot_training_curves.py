import os

import matplotlib.pyplot as plt
import numpy as np

from brever.modelmanagement import ModelFilterArgParser, find_model


def smooth(data, sigma=50):
    df = np.subtract.outer(np.arange(len(data)), np.arange(len(data)))
    filtering_mat = np.exp(-0.5*(df/sigma)**2)/(sigma*(2*np.pi)**0.5)
    filtering_mat = np.tril(filtering_mat)
    filtering_mat /= filtering_mat.sum(axis=1, keepdims=True)
    return filtering_mat@data


def main(**kwargs):
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')

    model_ids = find_model(**kwargs)

    plt.figure(figsize=(16, 8))
    for i, model_id in enumerate(model_ids):
        path = os.path.join('models', model_id, 'train_losses.npy')
        data = np.load(path)
        l, = plt.plot(data, label=f'{model_id[:3]}...')
        path = os.path.join('models', model_id, 'val_losses.npy')
        data = np.load(path)
        plt.plot(data, '--', color=l.get_color())
    plt.legend(ncol=10)
    plt.grid()

    plt.figure(figsize=(16, 8))
    for i, model_id in enumerate(model_ids):
        print(model_id)
        path = os.path.join('models', model_id, 'train_losses.npy')
        data = np.load(path)
        data = smooth(data)
        l, = plt.plot(data, label=f'{model_id[:3]}...')
        path = os.path.join('models', model_id, 'val_losses.npy')
        data = np.load(path)
        data = smooth(data)
        plt.plot(data, '--', color=l.get_color())
    plt.legend(ncol=10)
    plt.grid()

    plt.figure(figsize=(16, 8))
    for i, model_id in enumerate(model_ids):
        path = os.path.join('models', model_id, 'train_losses.npy')
        data = np.load(path)
        data = smooth(data)
        slope = np.zeros(len(data))
        strip = 100
        for i in range(len(data)-strip):
            x = np.arange(strip)
            y = data[i:i+strip]
            slope[i+strip] = np.sum((x - x.mean())*(y - y.mean()))/np.sum((x - x.mean())*(x - x.mean()))
        plt.plot(np.log10(abs(slope)))
    plt.grid()

    plt.show()


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='plot training curves')
    filter_args, _ = parser.parse_args()
    main(**vars(filter_args))
