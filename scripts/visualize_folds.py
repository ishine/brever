import itertools

import matplotlib.pyplot as plt
import numpy as np


alpha = 0.5
red = [1, 0, 0, alpha]
green = [0, 1, 0, alpha]


def init_fig():
    fig, axes = plt.subplots(7, 5, subplot_kw={'projection': '3d'})
    axes = iter(axes.flatten())
    return fig, axes


def format_ax(ax):
    ax.set_xticks(np.linspace(0, 5, 6))
    ax.set_yticks(np.linspace(0, 5, 6))
    ax.set_zticks(np.linspace(0, 5, 6))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def complement(idx_list):
    return [i for i in range(5) if i not in idx_list]


def build_test_index(index, dims):
    test_index = [complement(index[dim]) for dim in range(3)]
    for dim in dims:
        test_index[dim] = index[dim]
    return test_index


def train_voxels(ax, index):
    data = np.zeros((5, 5, 5), dtype=bool)
    for ii, jj, kk in itertools.product(*index):
        data[ii, jj, kk] = 1
    ax.voxels(data, facecolors=red)


def test_voxels(ax, index, dims):
    test_index = build_test_index(index, dims)
    data = np.zeros((5, 5, 5), dtype=bool)
    for ii, jj, kk in itertools.product(*test_index):
        data[ii, jj, kk] = 1
    ax.voxels(data, facecolors=green)


def main(index_func):
    fig, axes = init_fig()
    for ndim in range(3):
        for dims in itertools.combinations(range(3), ndim):
            for i in range(5):
                ax = next(axes)
                index = index_func(i, dims)
                train_voxels(ax, index)
                test_voxels(ax, index, dims)
                format_ax(ax)


def n_eq_one(i, dims):
    return [[i], [i], [i]]


def n_eq_four(i, dims):
    return [complement([i])]*3


def n_eq_one_old(i, dims):
    index = [[i], [i], [i]]
    for dim in dims:
        index[dim] = [0]
    return index


def n_eq_four_old(i, dims):
    index = [complement([i])]*3
    for dim in dims:
        index[dim] = [0]
    return index


if __name__ == '__main__':
    main(n_eq_one)
    main(n_eq_four)
    main(n_eq_one_old)
    main(n_eq_four_old)
    plt.show()
