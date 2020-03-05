import sys
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from brever.utils import standardize


# input paths
try:
    basename = sys.argv[1]
except IndexError:
    basename = 'temp'
dataset_path = 'data/datasets/%s.hdf5' % basename
pipes_path = 'data/datasets/%s.pkl' % basename

# load pipes
with open(pipes_path, 'rb') as f:
    pipes = pickle.load(f)

# show pipes
for pipe in pipes.values():
    print(pipe, end='\n\n')

# load features and labels
with h5py.File(dataset_path, 'r') as f:
    features = f['features'][:]
    labels = f['labels'][:]
    indices = f.attrs['indices']

# standardize features
global_standardization = False
if global_standardization:
    features = standardize(features)
else:
    for i_start, i_end in indices:
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
feature_names = pipes['featureExtractor'].features
feature_indices = pipes['featureExtractor'].indices
axes[0].set_yticks([index[0] for index in feature_indices])
axes[0].set_yticklabels([])
axes[0].set_yticks(np.mean(feature_indices, axis=1), minor=True)
axes[0].set_yticklabels(feature_names, minor=True)
axes[0].tick_params(axis='y', which='minor', length=0)
plt.tight_layout()
plt.show()
