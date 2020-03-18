import os
import logging
import time
import argparse
import pprint
import json
import pickle
import random

import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

from brever.config import defaults
from brever.classes import Standardizer
from brever.classes import (Filterbank, Framer, FeatureExtractor,
                            LabelExtractor, RandomMixtureMaker, UnitRMSScaler)


def main(input_config):
    with open(input_config, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)
    output_dir = os.path.dirname(input_config)

    # redirect logger
    logger = logging.getLogger()
    for i in reversed(range(len(logger.handlers))):
        logger.removeHandler(logger.handlers[i])
    logfile = os.path.join(output_dir, 'log.txt')
    filehandler = logging.FileHandler(logfile, mode='w')
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logging.info('\n' + pprint.pformat({'PRE': config.PRE.todict()}))

    # seed for reproducibility
    random.seed(0)

    # mixture maker
    randomMixtureMaker = RandomMixtureMaker(
        rooms=config.PRE.MIXTURES.RANDOM.ROOMS,
        angles_target=range(
            config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.MIN,
            config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.MAX + 1,
            config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.STEP,
        ),
        angles_directional=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.MAX + 1,
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.STEP,
        ),
        snrs=range(
            config.PRE.MIXTURES.RANDOM.TARGET.SNR.MIN,
            config.PRE.MIXTURES.RANDOM.TARGET.SNR.MAX + 1,
        ),
        snrs_directional_to_diffuse=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.SNR.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.SNR.MAX + 1,
        ),
        types_directional=config.PRE.MIXTURES.RANDOM.SOURCES.TYPES,
        types_diffuse=config.PRE.MIXTURES.RANDOM.DIFFUSE.TYPES,
        n_directional_sources=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER.MAX + 1,
        ),
        padding=config.PRE.MIXTURES.PADDING,
        reflection_boundary=config.PRE.MIXTURES.REFLECTIONBOUNDARY,
        fs=config.PRE.FS,
        noise_file_lims=config.PRE.MIXTURES.FILELIMITS.NOISE,
        target_file_lims=config.PRE.MIXTURES.FILELIMITS.TARGET,
        rms_jitter_dB=range(
            config.PRE.MIXTURES.RANDOM.RMSDB.MIN,
            config.PRE.MIXTURES.RANDOM.RMSDB.MAX + 1,
        ),
        surrey_dirpath=config.PRE.MIXTURES.PATH.SURREY,
        timit_dirpath=config.PRE.MIXTURES.PATH.TIMIT,
        dcase_dirpath=config.PRE.MIXTURES.PATH.DCASE,
    )

    # scaler
    scaler = UnitRMSScaler(
        active=config.PRE.SCALERMS,
    )

    # filterbank
    filterbank = Filterbank(
        kind=config.PRE.FILTERBANK.KIND,
        n_filters=config.PRE.FILTERBANK.NFILTERS,
        f_min=config.PRE.FILTERBANK.FMIN,
        f_max=config.PRE.FILTERBANK.FMAX,
        fs=config.PRE.FS,
        order=config.PRE.FILTERBANK.ORDER,
    )

    # framer
    framer = Framer(
        frame_length=config.PRE.FRAMER.FRAMELENGTH,
        hop_length=config.PRE.FRAMER.HOPLENGTH,
        window=config.PRE.FRAMER.WINDOW,
        center=config.PRE.FRAMER.CENTER,
    )

    # feature extractor
    featureExtractor = FeatureExtractor(
        features=config.PRE.FEATURES,
    )

    # label extractor
    labelExtractor = LabelExtractor(
        label=config.PRE.LABEL,
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
    for i in range(config.PRE.MIXTURES.NUMBER):

        # estimate time remaining and show progress
        if i == 0:
            logging.info((f'Processing mixture '
                          f'{i+1}/{config.PRE.MIXTURES.NUMBER}...'))
        else:
            etr = (config.PRE.MIXTURES.NUMBER-i)*total_time/i
            logging.info((f'Processing mixture '
                          f'{i+1}/{config.PRE.MIXTURES.NUMBER}... '
                          f'ETR: {int(etr/60)} m {int(etr%60)} s'))

        # make mixture and save
        components, metadata = randomMixtureMaker.make()
        mixture, foreground, background = components
        if config.PRE.MIXTURES.SAVE:
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
    datasets_output_path = os.path.join(output_dir, 'dataset.hdf5')
    with h5py.File(datasets_output_path, 'w') as f:
        f.create_dataset('features', data=features)
        f.create_dataset('labels', data=labels)
        if config.PRE.MIXTURES.SAVE:
            f.create_dataset('mixtures', data=mixtures,
                             dtype=h5py.vlen_dtype(float))
            f.create_dataset('foregrounds', data=foregrounds,
                             dtype=h5py.vlen_dtype(float))
            f.create_dataset('backgrounds', data=backgrounds,
                             dtype=h5py.vlen_dtype(float))

    # save mixtures metadata
    metadatas_output_path = os.path.join(output_dir, 'mixture_info.json')
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

    # plot a small sample of the dataset
    def plot(x, y, filename):
        gridspec_kw = {
            'height_ratios': [x.shape[1], y.shape[1]],
        }
        fig, axes = plt.subplots(2, 1, gridspec_kw=gridspec_kw)

        ax = axes[0]
        pos = ax.imshow(x.T, aspect='auto', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=.1, pad=0.1)
        fig.colorbar(pos, cax=cax)
        ax.set_title('features')
        cmin, cmax = np.quantile(x, [0.05, 0.95])
        ax.get_images()[0].set_clim(cmin, cmax)
        yticks_major = [index[0] for index in featureExtractor.indices]
        yticks_major.append(featureExtractor.indices[-1][1])
        axes[0].set_yticks(yticks_major)
        axes[0].set_yticklabels(yticks_major)
        yticks_minor = np.mean(featureExtractor.indices, axis=1)
        axes[0].set_yticks(yticks_minor, minor=True)
        axes[0].set_yticklabels(featureExtractor.features, minor=True)
        axes[0].tick_params(axis='y', which='minor', length=0)

        ax = axes[1]
        pos = ax.imshow(y.T, aspect='auto', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=.1, pad=0.1)
        fig.colorbar(pos, cax=cax)
        ax.set_title('labels')
        ax.get_images()[0].set_clim(0, 1)
        yticks_major = [0, y.shape[1]]
        axes[1].set_yticks(yticks_major)
        axes[1].set_yticklabels(yticks_major)
        yticks_minor = [y.shape[1]/2]
        axes[1].set_yticks(yticks_minor, minor=True)
        axes[1].set_yticklabels([labelExtractor.label], minor=True)
        axes[1].tick_params(axis='y', which='minor', length=0)

        peek_output_path = os.path.join(output_dir, filename)
        fig.tight_layout()
        fig.savefig(peek_output_path)

    if config.PRE.PLOT.ON:
        x = features[:config.PRE.PLOT.NSAMPLES]
        y = labels[:config.PRE.PLOT.NSAMPLES]
        plot(x, y, 'peek.png')

        standardizer = Standardizer()
        standardizer.fit(features)
        x = standardizer.transform(x)
        plot(x, y, 'peek_standardized.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset.')
    parser.add_argument('-i', '--input',
                        help=('Input YAML file.'))
    parser.add_argument('--all',
                        help=('Create all available datasets'),
                        action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
    )

    if args.all:
        datasets_path = 'data\\processed'
        for root, folders, files in os.walk(datasets_path):
            files = [file for file in files if file.endswith('.yaml')]
            if len(files) > 1:
                raise ValueError(f'More than one YAML file in {root}.')
            for file in files:
                main(os.path.join(root, file))

    else:
        main(args.input)
