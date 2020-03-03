import sys
import pickle
import random
import h5py
import numpy as np
import time

from brever.classes import (Filterbank, Framer, FeatureExtractor,
                            LabelExtractor, RandomMixtureMaker)


# output paths
try:
    output_basename = sys.argv[1]
except IndexError:
    output_basename = 'temp'
main_output_path = 'data/datasets/%s.hdf5' % output_basename
pipes_output_path = 'data/datasets/%s_pipes.pkl' % output_basename

# set random state for reproducibility
random.seed(42)

# sampling frequency
fs = 16000

# filterbank
filterbank = Filterbank(
    kind='mel',
    n_filters=64,
    f_min=50,
    f_max=8000,
    fs=fs,
    order=4,
)

# framer
framer = Framer(
    frame_length=512,
    hop_length=256,
    window='hann',
    center=False,
)

# feature extractor
featureExtractor = FeatureExtractor(
    features=[
        'ild',
        'itd_ic',
        'mfcc',
        'pdf',
    ]
)

# label extractor
labelExtractor = LabelExtractor()

# mixture maker
randomMixtureMaker = RandomMixtureMaker(
    rooms=['surrey_room_a'],
    angles=[0],
    snrs=range(-5, 16),
    padding=0,
    reflection_boundary=10e-3,
    max_itd=1e-3,
    fs=fs,
)

# number of mixtures
n_mixtures = 2

# main loop
features = []
labels = []
mixtures = []
metadata = {
    'snrs': [],
    'filenames': [],
    'indices': [],
}
i_start = 0
total_time = 0
start_time = time.time()
for i in range(n_mixtures):

    # estimate time remaining and show progress
    if i == 0:
        print('Processing mixture %i/%i...' % (i+1, n_mixtures))
    else:
        etr = (n_mixtures-i)*total_time/i
        print(('Processing mixture %i/%i... ETR: %i min %i s'
               % (i+1, n_mixtures, etr//60, etr % 60)))

    # make mixture
    mixtureObject = randomMixtureMaker.make()
    mixtures.append(mixtureObject.mix)

    # apply filterbank
    mix = filterbank.filt(mixtureObject.mix)
    foreground = filterbank.filt(mixtureObject.foreground)
    background = filterbank.filt(mixtureObject.background)

    # frame
    mix = framer.frame(mix)
    foreground = framer.frame(foreground)
    background = framer.frame(background)

    # extract features
    features.append(featureExtractor.run(mix))

    # extract labels
    labels.append(labelExtractor.run(foreground, background))

    # save metadata
    metadata['snrs'].append(mixtureObject.snr)
    metadata['filenames'].append(mixtureObject.filename)
    i_end = i_start + len(mix)
    metadata['indices'].append((i_start, i_end))
    i_start = i_end

    # update time spent
    total_time = time.time() - start_time

# concatenate features and labels
features = np.vstack(features)
labels = np.vstack(labels)

# save features, labels, mixtures and mixture metadata
with h5py.File(main_output_path, 'w') as f:
    f.create_dataset('features', data=features)
    f.create_dataset('labels', data=labels)
    f.create_dataset('mixtures', data=mixtures,
                     dtype=h5py.vlen_dtype(mixtures[0].dtype))
    f.attrs.update(metadata)

# save pipes
pipes = {
    'filterbank': filterbank,
    'framer': framer,
    'featureExtractor': featureExtractor,
    'labelExtractor': labelExtractor,
    'randomMixtureMaker': randomMixtureMaker,
}
with open(pipes_output_path, 'wb') as f:
    pickle.dump(pipes, f)
