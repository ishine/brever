import random

import matplotlib.pyplot as plt
import numpy as np
import json

from brever.classes import (DefaultRandomMixtureMaker, DefaultFilterbank,
                            DefaultFramer)
from brever.display import (plot_waveform, plot_spectrogram, share_ylim,
                            share_clim)


# seed for reproducibility
random.seed(12)

# initialize classes
rmm = DefaultRandomMixtureMaker()
filterbank = DefaultFilterbank()
framer = DefaultFramer()

# make a random mixture and print metadata
mixObj, metadata = rmm.make()
print(json.dumps(metadata, indent=4))

# plot waveform and spectrogram for each component
attributes = ['mixture', 'target', 'noise']
fig, axes = plt.subplots(len(attributes), 2)
for i, attribute in enumerate(attributes):
    x = getattr(mixObj, attribute)

    # average across left and right channels
    x_mono = x.mean(axis=1)

    # plot waveform
    plot_waveform(
        x_mono,
        ax=axes[i, 0],
        fs=rmm.fs,
        set_kw={'title': attribute},
    )

    # filter
    x_filtered = filterbank.filt(x_mono)

    # frame
    x_framed = framer.frame(x_filtered)

    # average energy in each frame
    X = (x_framed**2).mean(axis=1)

    # convert to dB
    X_dB = 20*np.log10(X + 1e-10)

    # plot spectrogram
    plot_spectrogram(
        X_dB,
        ax=axes[i, 1],
        fs=rmm.fs,
        hop_length=framer.hop_length,
        f=filterbank.fc,
        set_kw={'title': attribute},
    )

# match limits
share_ylim(axes[:, 0], center_zero=True)
share_clim(axes[:, 1])

plt.tight_layout()
plt.show()
