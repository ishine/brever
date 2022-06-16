import itertools

import matplotlib.pyplot as plt
import numpy as np


alpha = 0.5
red = [1, 0, 0, alpha]
green = [0, 1, 0, alpha]


fig, axes = plt.subplots(7, 5, subplot_kw={'projection': '3d'})
axes = iter(axes.flatten())

for ndim in range(3):

    for dims in itertools.combinations(range(3), ndim):
        other_dims = [i for i in range(3) if i not in dims]

        for i in range(5):

            ax = next(axes)

            data = np.zeros((5, 5, 5), dtype=bool)
            data[i, i, i] = 1
            ax.voxels(data, facecolors=red)

            data = np.zeros((5, 5, 5), dtype=bool)
            for ii, jj, kk in itertools.product(range(5), repeat=3):
                if all([ii, jj, kk][dim] == i for dim in dims):
                    if all([ii, jj, kk][dim] != i for dim in other_dims):
                        data[ii, jj, kk] = 1

            ax.voxels(data, facecolors=green)

            ax.set_xticks(np.linspace(0, 5, 6))
            ax.set_yticks(np.linspace(0, 5, 6))
            ax.set_zticks(np.linspace(0, 5, 6))

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])


fig, axes = plt.subplots(7, 5, subplot_kw={'projection': '3d'})
axes = iter(axes.flatten())

for ndim in range(3):

    for dims in itertools.combinations(range(3), ndim):
        other_dims = [i for i in range(3) if i not in dims]

        for i in range(5):

            ax = next(axes)

            data = np.zeros((5, 5, 5), dtype=bool)
            indices = [ii for ii in range(5) if ii != i]
            for ii, jj, kk in itertools.product(indices, repeat=3):
                data[ii, jj, kk] = 1
            ax.voxels(data, facecolors=red)

            data = np.zeros((5, 5, 5), dtype=bool)
            for ii, jj, kk in itertools.product(range(5), repeat=3):
                if all([ii, jj, kk][dim] in indices for dim in dims):
                    if all([ii, jj, kk][dim] not in indices for dim in other_dims):
                        data[ii, jj, kk] = 1

            ax.voxels(data, facecolors=green)

            ax.set_xticks(np.linspace(0, 5, 6))
            ax.set_yticks(np.linspace(0, 5, 6))
            ax.set_zticks(np.linspace(0, 5, 6))

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

plt.show()
