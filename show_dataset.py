import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pprint import pprint


dataset_path = sys.argv[1]


with h5py.File(dataset_path, 'r') as f:
    features = f['features'][:]
    labels = f['labels'][:]
    metadata = dict(f.attrs.items())

for key, value in metadata.items():
    if isinstance(value, np.ndarray):
        metadata[key] = value.tolist()
pprint(metadata)

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
