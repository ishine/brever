import os

import matplotlib
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
    model_ids = find_model(**kwargs)

    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    colors = cmap(np.linspace(0, 1, len(model_ids)))

    plt.figure(figsize=(16, 8))
    for i, model_id in enumerate(model_ids):
        print(model_id)
        path = os.path.join('models', model_id, 'train_losses.npy')
        data = np.load(path)
        data = smooth(data)
        plt.plot(data, color=colors[i], label=f'{model_id[:3]}...')
        path = os.path.join('models', model_id, 'val_losses.npy')
        data = np.load(path)
        data = smooth(data)
        plt.plot(data, '--', color=colors[i])
    plt.legend(ncol=10)

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
        plt.ylim(-6, -4)
    plt.grid()

    plt.show()


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='plot training curves')
    filter_args, _ = parser.parse_args()
    main(**vars(filter_args))
