import sys
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pprint import pprint

from brever.utils import standardize


# input paths
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
    pipes_path = sys.argv[2]
else:
    basename = 'temp'
    dataset_path = 'data/datasets/%s.hdf5' % basename
    pipes_path = 'data/datasets/%s.pkl' % basename

# load pipes
with open(pipes_path, 'rb') as f:
    pipes = pickle.load(f)

# show pipes
for pipe in pipes.values():
    print(pipe)

# load features, labels, mixtures and mixture metadata
with h5py.File(dataset_path, 'r') as f:
    features = f['features'][:]
    labels = f['labels'][:]
    metadata = dict(f.attrs.items())

# show mixture metadata
for key, value in metadata.items():
    if isinstance(value, np.ndarray):
        metadata[key] = value.tolist()
pprint(metadata)

# standardize features
global_standardization = False
if global_standardization:
    features = standardize(features)
else:
    for i_start, i_end in metadata['indices']:
        features[i_start:i_end] = standardize(features[i_start:i_end])

# plot
fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
for i, name in enumerate(['features', 'labels']):
    x = locals()[name].T
    pos = axes[i].imshow(x, aspect='auto', origin='lower')
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes("right", size="1%", pad=0.1)
    fig.colorbar(pos, cax=cax)
    axes[i].set_title(name)
cmin, cmax = np.quantile(features, [0.05, 0.95])
axes[0].get_images()[0].set_clim(cmin, cmax)
axes[1].get_images()[0].set_clim(0, 1)
plt.tight_layout()
plt.show()
