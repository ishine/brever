import h5py
import matplotlib.pyplot as plt


dataset_path = 'data/datasets/temp.h5'

n_frames = 200

with h5py.File(dataset_path, 'r') as f:
    features = f['features'][:n_frames]
    labels = f['labels'][:n_frames]

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
pos = ax.imshow(features.T, aspect='auto', origin='lower', vmin=-2, vmax=2)
fig.colorbar(pos, ax=ax)
ax = fig.add_subplot(2, 1, 2)
pos = ax.imshow(labels.T, aspect='auto', origin='lower')
fig.colorbar(pos, ax=ax)

plt.tight_layout()
plt.show()
