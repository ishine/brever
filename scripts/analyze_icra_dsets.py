import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import brever.pytorchtools as bptt
import brever.modelmanagement as bmm
from brever.config import defaults

import ecol
import ecol.pls


def inv_sigmoid(x):
    eps = np.finfo(float).eps
    return np.log((x + eps)/(1 - x + eps))


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

    scores = []
    scores_pls = []

    fig_dist, ax_dist = plt.subplots()

    for path in paths:

        print(path)

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

        ax_dist.hist(
            inv_sigmoid(labels.flatten()),
            bins=100,
            label=os.path.basename(path),
            alpha=0.33,
        )

        labels_cont = labels
        labels_bin = labels > 0.5

        n = len(features)
        m = n//10
        i_start = 0
        features = features[i_start:i_start+m]
        labels_cont = labels_cont[i_start:i_start+m]
        labels_bin = labels_bin[i_start:i_start+m]

        print('dist mat...')
        dist_mat = pdist(features)
        dist_mat = squareform(dist_mat)

        print('N1...')
        n1 = ecol.N1(features, labels_bin, dist_mat=dist_mat)
        print('N2...')
        n2 = ecol.N2(features, labels_bin, dist_mat=dist_mat)
        print('N3...')
        n3 = ecol.N3(features, labels_bin, dist_mat=dist_mat)
        # print('N4...')
        # n4 = ecol.N4(features, labels_bin)
        # print('T1...')
        # t1 = ecol.T1(features, labels_bin, dist_mat=dist_mat)
        print('T2...')
        t2 = ecol.T2(features)
        print('T3...')
        t3 = ecol.T3(features)
        print('T4...')
        t4 = ecol.T4(features)
        print('CB...')
        cb = 1 - labels_bin.mean()

        print('PLS...')
        try:
            pvex, pvey = ecol.pls.pls(features, labels_cont, features.shape[1])
        except np.linalg.LinAlgError:
            pls_x = 0
            pls_y = 0
        else:
            n_components = np.argmax(np.cumsum(pvex) >= 0.95) + 1
            pls_x = n_components/features.shape[1]
            pls_y = pvey.sum()

        scores.append((n1, n2, n3, t2, t3, t4, cb))
        scores_pls.append((t4, pls_x, pls_y))

    scores = np.asarray(scores)
    xticks = np.arange(len(paths))
    xticklabels = [os.path.basename(path) for path in paths]
    fig, ax = plt.subplots()
    n, m = scores.shape
    width = 1/(m+1)
    labels = ['n1', 'n2', 'n3', 't2', 't3', 't4', 'cb']
    for j in range(m):
        offset = (j - (m-1)/2)*width
        x = np.arange(n) + offset
        ax.bar(x, scores[:, j], width=width, label=labels[j])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.legend()

    scores_pls = np.asarray(scores_pls)
    xticks = np.arange(len(paths))
    xticklabels = [os.path.basename(path) for path in paths]
    fig, ax = plt.subplots()
    n, m = scores_pls.shape
    width = 1/(m+1)
    labels = ['pca', 'pls_x', 'pls_y']
    for j in range(m):
        offset = (j - (m-1)/2)*width
        x = np.arange(n) + offset
        ax.bar(x, scores_pls[:, j], width=width, label=labels[j])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.legend()

    ax_dist.legend()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
