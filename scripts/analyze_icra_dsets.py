import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import brever.pytorchtools as bptt
import brever.modelmanagement as bmm
from brever.config import defaults
from brever.utils import pca


def main(args):
    paths = [
        'data/processed/test/icra_01',
        'data/processed/test/icra_02',
        'data/processed/test/icra_03',
        'data/processed/test/icra_04',
        'data/processed/test/icra_05',
        'data/processed/test/icra_06',
        'data/processed/test/icra_07',
        'data/processed/test/icra_08',
        'data/processed/test/icra_09',
    ]

    scores_pca = []
    scores_class = []

    fig_dist, ax_dist = plt.subplots()

    for path in paths:

        # load config file
        config = defaults()
        config_file = os.path.join(path, 'config.yaml')
        config.update(bmm.read_yaml(config_file))

        dset = bptt.H5Dataset(
            dirpath=path,
            features=config.POST.FEATURES,
            labels=config.POST.LABELS,
            load=config.POST.LOAD,
            stack=5,
            decimation=1,  # there must not be decimation during testing
            dct_toggle=config.POST.DCT.ON,
            n_dct=config.POST.DCT.NCOEFF,
            prestack=config.POST.PRESTACK,
        )

        features, labels = dset[:]

        components, ve, means = pca(features, fve=0.95)
        scores_pca.append(components.shape[1])

        scores_class.append((labels < 0.5).mean())

        ax_dist.hist(
            labels.flatten(),
            bins=100,
            label=os.path.basename(path),
            alpha=0.33,
        )

    xticks = np.arange(len(paths))
    xticklabels = [os.path.basename(path) for path in paths]
    fig, ax = plt.subplots()
    l1 = ax.bar(
        xticks-0.33/2,
        scores_pca,
        width=0.33,
        color='tab:blue',
    )
    ax.set_ylabel('number of components')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax = ax.twinx()
    l2 = ax.bar(
        xticks+0.33/2,
        scores_class,
        width=0.33,
        color='tab:orange',
    )
    ax.set_ylabel('fraction of zeros')
    ax.legend([l1, l2], ['pca', 'labels'])

    ax_dist.legend()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
