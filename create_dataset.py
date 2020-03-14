import argparse
import pickle
import random
import h5py
import json
import numpy as np
import time
import os

from brever.classes import (Filterbank, Framer, FeatureExtractor,
                            LabelExtractor, RandomMixtureMaker, UnitRMSScaler)
from brever import config


parser = argparse.ArgumentParser(description='Create a dataset.')
parser.add_argument('-i', '--input',
                    help=('Input YAML file. If none is provided, default '
                          'settings will be used.'))
parser.add_argument('-o', '--output',
                    help=('Custom output directory. If it does not exist, it '
                          'is created. If none is provided, the outputs are '
                          'created next to the input config file.'))
args = parser.parse_args()

if not args.input and not args.output:
    raise ValueError(('The output directory must be specified if no input '
                      'YAML file is given. Use -h for help.'))

if args.input:
    config.update(args.input)
    output_dir = os.path.dirname(args.input)

if args.output:
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    elif not os.path.isdir(args.output):
        raise ValueError(('The specified output path points to an already '
                          'existing file.'))
    output_dir = args.output


# set random state for reproducibility
random.seed(0)

# mixture maker
randomMixtureMaker = RandomMixtureMaker(
    rooms=config.MIXTURES.RANDOM.ROOMS,
    angles_target=range(
        config.MIXTURES.RANDOM.TARGET.ANGLE.MIN,
        config.MIXTURES.RANDOM.TARGET.ANGLE.MAX + 1,
        config.MIXTURES.RANDOM.TARGET.ANGLE.STEP,
    ),
    angles_directional=range(
        config.MIXTURES.RANDOM.SOURCES.ANGLE.MIN,
        config.MIXTURES.RANDOM.SOURCES.ANGLE.MAX + 1,
        config.MIXTURES.RANDOM.SOURCES.ANGLE.STEP,
    ),
    snrs=range(
        config.MIXTURES.RANDOM.TARGET.SNR.MIN,
        config.MIXTURES.RANDOM.TARGET.SNR.MAX + 1,
    ),
    snrs_directional_to_diffuse=range(
        config.MIXTURES.RANDOM.SOURCES.SNR.MIN,
        config.MIXTURES.RANDOM.SOURCES.SNR.MAX + 1,
    ),
    types_directional=config.MIXTURES.RANDOM.SOURCES.TYPES,
    types_diffuse=config.MIXTURES.RANDOM.DIFFUSE.TYPES,
    n_directional_sources=range(
        config.MIXTURES.RANDOM.SOURCES.NUMBER.MIN,
        config.MIXTURES.RANDOM.SOURCES.NUMBER.MAX + 1,
    ),
    padding=config.MIXTURES.PADDING,
    reflection_boundary=config.MIXTURES.REFLECTIONBOUNDARY,
    fs=config.FS,
    noise_file_lims=config.MIXTURES.FILELIMITS.NOISE,
    target_file_lims=config.MIXTURES.FILELIMITS.TARGET,
    rms_jitter_dB=range(
        config.MIXTURES.RANDOM.RMSDB.MIN,
        config.MIXTURES.RANDOM.RMSDB.MAX + 1,
    ),
    surrey_dirpath=config.PATH.SURREY,
    timit_dirpath=config.PATH.TIMIT,
    dcase_dirpath=config.PATH.DCASE,
)

# scaler
scaler = UnitRMSScaler(
    active=config.MIXTURES.SCALERMS,
)

# filterbank
filterbank = Filterbank(
    kind=config.FILTERBANK.KIND,
    n_filters=config.FILTERBANK.NFILTERS,
    f_min=config.FILTERBANK.FMIN,
    f_max=config.FILTERBANK.FMAX,
    fs=config.FILTERBANK.FS,
    order=config.FILTERBANK.ORDER,
)

# framer
framer = Framer(
    frame_length=config.FRAMER.FRAMELENGTH,
    hop_length=config.FRAMER.HOPLENGTH,
    window=config.FRAMER.WINDOW,
    center=config.FRAMER.CENTER,
)

# feature extractor
featureExtractor = FeatureExtractor(
    features=config.FEATURES,
)

# label extractor
labelExtractor = LabelExtractor(
    label=config.LABEL,
)

# main loop
features = []
labels = []
mixtures = []
foregrounds = []
backgrounds = []
metadatas = []
i_start = 0
total_time = 0
start_time = time.time()
for i in range(config.MIXTURES.NUMBER):

    # estimate time remaining and show progress
    if i == 0:
        print('Processing mixture %i/%i...' % (i+1, config.MIXTURES.NUMBER))
    else:
        etr = (config.MIXTURES.NUMBER-i)*total_time/i
        print(('Processing mixture %i/%i... ETR: %i min %i s'
               % (i+1, config.MIXTURES.NUMBER, etr//60, etr % 60)))

    # make mixture and save
    components, metadata = randomMixtureMaker.make()
    mixture, foreground, background = components
    if config.MIXTURES.SAVE:
        mixtures.append(mixture.flatten())
        foregrounds.append(foreground.flatten())
        backgrounds.append(background.flatten())

    # scale signal
    scaler.fit(mixture)
    mixture = scaler.scale(mixture)
    foreground = scaler.scale(foreground)
    background = scaler.scale(background)
    scaler.__init__(scaler.active)

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
    i_start = i_end
    metadata['dataset_indices'] = (i_start, i_end)
    metadatas.append(metadata)

    # update time spent
    total_time = time.time() - start_time

# concatenate features and labels
features = np.vstack(features)
labels = np.vstack(labels)

# save datasets
datasets_output_path = os.path.join(output_dir, 'datasets.hdf5')
with h5py.File(datasets_output_path, 'w') as f:
    f.create_dataset('features', data=features)
    f.create_dataset('labels', data=labels)
    if config.MIXTURES.SAVE:
        f.create_dataset('mixtures', data=mixtures,
                         dtype=h5py.vlen_dtype(float))
        f.create_dataset('foregrounds', data=foregrounds,
                         dtype=h5py.vlen_dtype(float))
        f.create_dataset('backgrounds', data=backgrounds,
                         dtype=h5py.vlen_dtype(float))

# save mixtures metadata
metadatas_output_path = os.path.join(output_dir, 'metadatas.json')
with open(metadatas_output_path, 'w') as f:
    json.dump(metadatas, f)

# save pipes
pipes_output_path = os.path.join(output_dir, 'pipes.pkl')
with open(pipes_output_path, 'wb') as f:
    pipes = {
        'scaler': scaler,
        'randomMixtureMaker': randomMixtureMaker,
        'filterbank': filterbank,
        'framer': framer,
        'featureExtractor': featureExtractor,
        'labelExtractor': labelExtractor,
    }
    pickle.dump(pipes, f)
