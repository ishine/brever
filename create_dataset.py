import os
import random
import h5py
import numpy as np
import time

import brever


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
    brirs.append(brever.load(filepath))

# load and resample sentences
print('Loading and resampling targets...')
sentences_dirpath = 'C:/Workspace/brever/data/audio/EMIME'
sentences = []
for root, dirs, files in os.walk(sentences_dirpath):
    for filename in files:
        if filename.endswith('.wav'):
            filepath = os.path.join(root, filename)
            sentences.append(brever.load(filepath))

# talker fixed at 0 degrees for now
brir_filepath = 'data/brirs/SURREY/Room_A/16kHz/CortexBRIR_0_32s_0deg_16k.wav'
brir = brever.load(brir_filepath)

# mixture parameters
n_mixtures = 100
snrs = range(-5, 16)
padding = round(0*fs)

# features to calculate
feature_list = ['ild', 'itd_ic']

# filterbank to use, either 'gammatone' or 'mel'
filter_type = 'gammatone'

# main loop
features = []
labels = []
times_spent = np.zeros(len(feature_list))
total_time_spent = 0
start_time = time.time()
for i in range(n_mixtures):

    # estimate time remaining and show progress
    if i == 0:
        print('Processing mixture %i/%i...' % (i+1, n_mixtures))
    else:
        etr = (n_mixtures-i)*total_time_spent/i
        print(('Processing mixture %i/%i... ETR: %i min %i s'
               % (i+1, n_mixtures, etr//60, etr % 60)))

    # make mixture
    sentence = random.choice(sentences)
    snr = random.choice(snrs)
    mix, _, target, _, _ = brever.mixture.make(sentence, brir, brirs, snr,
                                               padding)
    # extract features
    mix, _ = brever.filters.filt(mix, filter_type)
    mix = brever.utils.frame(mix)
    features_mix = []
    for i, feature_name in enumerate(feature_list):
        t = time.time()
        feature_func = getattr(brever.features, feature_name)
        features_mix.append(feature_func(mix, filtered=True,  framed=True))
        times_spent[i] += time.time() - t
    features.append(np.hstack(features_mix))

    # extract labels
    target, _ = brever.filters.filt(target, filter_type)
    target = brever.utils.frame(target)
    IRM = brever.labels.irm(target, mix, filtered=True, framed=True)
    labels.append(IRM)

    # update time spent
    total_time_spent = time.time() - start_time

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
