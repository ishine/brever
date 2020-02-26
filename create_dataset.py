import os
import random
import h5py
import numpy as np
import time

import brever


# set random state for reproducibility
random.seed(42)

# output dataset path
output_path = 'data/datasets/temp.hdf5'

# sampling frequency
fs = 16e3

# load brirs
print('Loading BRIRs...')
brir_dirpath = 'data/brirs/SURREY/Anechoic/16kHz'
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
            x = brever.load(filepath)
            item = {'filename': filename, 'x': x}
            sentences.append(item)

# talker fixed at 0 degrees for now
brir_filepath = 'data/brirs/SURREY/Anechoic/16kHz/CortexBRIR_0s_0deg_16k.wav'
brir = brever.load(brir_filepath)

# mixture parameters
n_mixtures = 5
snr_range = (-5, 15)
padding = round(0*fs)

# irm parameters  #TODO: add attenuation threshold
reflection_boundary = 10e-3

# features to calculate
# feature_list = ['ild', 'itd_ic', 'mfcc', 'pdf']
feature_list = ['ic', 'pdf', 'mfcc']

# filterbank to use, either 'gammatone' or 'mel'
filterbank = 'mel'

# main loop
features = []
labels = []
mixtures = []
metadata = {
    'n_mixtures': n_mixtures,
    'snr_range': snr_range,
    'padding': padding,
    'features': feature_list,
    'filterbank': filterbank,
    'brir_dirpath': brir_dirpath,
    'indices': [],
    'snrs': [],
    'filenames': [],
}
mixture_index_start = 0
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
    snr = np.random.randint(*snr_range)
    components = brever.mixture.make(sentence['x'], brir, brirs, snr, padding,
                                     reflection_boundary)
    mix, _, target_early, target_late, noise = components
    mixtures.append(mix)

    # extract features
    mix, _ = brever.filters.filt(mix, filterbank)
    mix = brever.utils.frame(mix)
    features_mix = []
    for i, feature_name in enumerate(feature_list):
        t = time.time()
        feature_func = getattr(brever.features, feature_name)
        features_mix.append(feature_func(mix, filtered=True,  framed=True))
        times_spent[i] += time.time() - t
    features.append(np.hstack(features_mix))

    # extract labels
    IRM = brever.labels.irm(target_early, target_late+noise,
                            filt_kwargs={'filter_type': filterbank})
    labels.append(IRM)

    # save metadata
    metadata['snrs'].append(snr)
    metadata['filenames'].append(sentence['filename'])
    mixture_index_end = mixture_index_start + len(mix)
    metadata['indices'].append((mixture_index_start, mixture_index_end))
    mixture_index_start = mixture_index_end

    # update time spent
    total_time_spent = time.time() - start_time

# print time spent
for feature_name, time_spent in zip(feature_list, times_spent):
    print(('Time spent calculating "%s": %i min %i s'
           % (feature_name, time_spent//60, time_spent % 60)))

# concatenate features and labels
features = np.vstack(features)
labels = np.vstack(labels)

# global standardization
features = brever.utils.standardize(features)

# save data
with h5py.File(output_path, 'w') as f:
    f.create_dataset('features', data=features)
    f.create_dataset('labels', data=labels)
    f.create_dataset('mixtures', data=mixtures,
                     dtype=h5py.vlen_dtype(mixtures[0].dtype))
    f.attrs.update(metadata)
