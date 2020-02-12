import os
import sys
import soundfile as sf
from resampy import resample
import random
import h5py
import numpy as np
import time

import brever
import brever.utils
import brever.features
import brever.mixture


# set random state for reproducibility
random.seed(0)

# output dataset path
output_path = 'data/datasets/temp.h5'

# sampling frequency
fs = 16e3

# load brirs
print('Loading BRIRs...')
brir_dirpath = 'data/brirs/SURREY/Room_A/16kHz'
brirs = []
for filename in os.listdir(brir_dirpath):
    filepath = os.path.join(brir_dirpath, filename)
    brir, brir_fs = sf.read(filepath)
    assert brir_fs == fs
    brirs.append(brir)

# load and resample sentences
print('Loading and resampling targets...')
sentences_dirpath = 'C:/Workspace/brever/data/audio/EMIME'
sentences = []
for root, dirs, files in os.walk(sentences_dirpath):
    for filename in files:
        if filename.endswith('.wav'):
            filepath = os.path.join(root, filename)
            sentence, fs_old = sf.read(filepath)
            sentence = resample(sentence, fs_old, fs)
            sentences.append(sentence)

# talker fixed at 0 degrees for now
brir_filepath = 'data/brirs/SURREY/Room_A/16kHz/CortexBRIR_0_32s_0deg_16k.wav'
brir, brir_fs = sf.read(brir_filepath)
assert brir_fs == fs

# anechoic brir correponding to target_brir
hrir_filepath = 'data/brirs/SURREY/Anechoic/16kHz/CortexBRIR_0s_0deg_16k.wav'
hrir, hrir_fs = sf.read(hrir_filepath)
assert hrir_fs == fs

# mixture parameters
snrs = range(-5, 16)
n_mixtures = 1
padding = round(0*fs)

# features to calculate
feature_list = ['ild', 'itd', 'ic']

# main loop
features = []
labels = []
times_spent = np.zeros(len(feature_list))
for i in range(n_mixtures):
    sys.stdout.write('\rProcessing mixture %i/%i...' % (i+1, n_mixtures))
    sentence = random.choice(sentences)
    snr = random.choice(snrs)
    # make mixture
    mix, target, noise = brever.mixture.mixture(sentence, brir, hrir, brirs,
                                                snr, padding)
    # extract features
    mixture_filt, _ = brever.gammatone_filt(mix)
    features_mix = []
    for i, feature_name in enumerate(feature_list):
        t = time.time()
        feature_func = getattr(brever.features, feature_name)
        features_mix.append(feature_func(mixture_filt))
        times_spent[i] += time.time() - t
    features.append(np.hstack(features_mix))

    # extract labels
    IRM = brever.irm(target, noise)
    labels.append(IRM)
print('')

# print time spent
for feature_name, time_spent in zip(feature_list, times_spent):
    print('Time spent calculating "%s": %.2f' % (feature_name, time_spent))

# concatenate features and labels
features = np.vstack(features)
labels = np.vstack(labels)

# global standardization
features = brever.utils.standardize(features)

# save data
with h5py.File(output_path, 'w') as f:
    f.create_dataset('features', data=features)
    f.create_dataset('labels', data=labels)
