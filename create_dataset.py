import sys
import pickle
import random
import h5py
import json
import numpy as np
import time

from brever.classes import (Filterbank, Framer, FeatureExtractor,
                            LabelExtractor, RandomMixtureMaker)


# output paths
try:
    output_basename = sys.argv[1]
except IndexError:
    output_basename = 'training'
datasets_output_path = 'data/datasets/%s.hdf5' % output_basename
pipes_output_path = 'data/datasets/%s.pkl' % output_basename
metadatas_output_path = 'data/datasets/%s.json' % output_basename

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
labelExtractor = LabelExtractor(
    label='irm'
)

# mixture maker
randomMixtureMaker = RandomMixtureMaker(
    rooms=['surrey_room_a'],
    angles_target=range(-90, 95, 5),
    angles_directional=range(-90, 95, 5),
    snrs=range(0, 16),
    snrs_directional_to_diffuse=range(-5, 6),
    types_directional=[
        'dcase_airport',
        'dcase_bus',
        'dcase_metro',
        'dcase_park',
        'dcase_public_square',
        'dcase_shopping_mall',
        'dcase_street_pedestrian',
        'dcase_street_traffic',
        'dcase_tram',
    ],
    types_diffuse=['noise_pink'],
    n_directional_sources=range(4),
    padding=0.5,
    reflection_boundary=50e-3,
    fs=fs,
    noise_file_lims=(0.0, 0.5),
    target_file_lims=(0.0, 0.5),
    rms_jitter_dB=range(-30, -10),
)

# number of mixtures
n_mixtures = 200

# main loop
features = []
labels = []
mixtures = []
foregrounds = []
backgrounds = []
metadatas = []
indices = []
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

    # make mixture and save
    components, metadata = randomMixtureMaker.make()
    mixture, foreground, background = components
    mixtures.append(mixture.flatten())
    foregrounds.append(foreground.flatten())
    backgrounds.append(background.flatten())
    metadatas.append(metadata)

    # apply filterbank
    mixture = filterbank.filt(mixture)
    foreground = filterbank.filt(foreground)
    background = filterbank.filt(background)

    # frame
    mixture = framer.frame(mixture)
    foreground = framer.frame(foreground)
    background = framer.frame(background)

    # extract features
    features.append(featureExtractor.run(mixture))

    # extract labels
    labels.append(labelExtractor.run(foreground, background))

    # save indices
    i_end = i_start + len(mixture)
    indices.append((i_start, i_end))
    i_start = i_end

    # update time spent
    total_time = time.time() - start_time

# concatenate features and labels
features = np.vstack(features)
labels = np.vstack(labels)

# save datasets
with h5py.File(datasets_output_path, 'w') as f:
    f.create_dataset('features', data=features)
    f.create_dataset('labels', data=labels)
    f.create_dataset('mixtures', data=mixtures,
                     dtype=h5py.vlen_dtype(float))
    f.create_dataset('foregrounds', data=foregrounds,
                     dtype=h5py.vlen_dtype(float))
    f.create_dataset('backgrounds', data=backgrounds,
                     dtype=h5py.vlen_dtype(float))
    f.attrs['indices'] = indices

# save mixtures metadata
with open(metadatas_output_path, 'w') as f:
    json.dump(metadatas, f)

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
