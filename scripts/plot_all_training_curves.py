import os
import argparse

import yaml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from brever.modelmanagement import ModelFilterArgParser, find_model


def main(**kwargs):
    model_ids = find_model(**kwargs)

    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    colors = cmap(np.linspace(0, 1, len(model_ids)))

    plt.figure(figsize=(16, 8))
    for i, model_id in enumerate(model_ids):
        print(model_id)
        path = os.path.join('models', model_id, 'train_losses.npy')
        data = np.load(path)
        plt.plot(data, color=colors[i], label=f'{model_id[:3]}...')
        path = os.path.join('models', model_id, 'val_losses.npy')
        data = np.load(path)
        plt.plot(data, '--', color=colors[i])
    plt.legend(ncol=10)

    plt.show()


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='plot training curves')
    args = parser.parse_args()
    main(**vars(args))
