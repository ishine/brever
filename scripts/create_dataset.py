import os
import argparse
import logging
import time
import pprint
import json
import pickle
import random
from glob import glob
import sys

import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import soundfile as sf

from brever.config import defaults
from brever.classes import (Standardizer, Filterbank, Framer, FeatureExtractor,
                            LabelExtractor, RandomMixtureMaker, UnitRMSScaler)


def main(dataset_dir, force):
    # check if dataset already exists
    datasets_output_path = os.path.join(dataset_dir, 'dataset.hdf5')
    if os.path.exists(datasets_output_path) and not force:
        logging.info('Dataset already exists')
        return

    config_file = os.path.join(dataset_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    # redirect logger
    logger = logging.getLogger()
    for i in reversed(range(len(logger.handlers))):
        logger.removeHandler(logger.handlers[i])
    logfile = os.path.join(dataset_dir, 'log.txt')
    filehandler = logging.FileHandler(logfile, mode='w')
    streamhandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logging.info(pprint.pformat({'PRE': config.PRE.to_dict()}))

    # seed for reproducibility
    if config.PRE.SEED.ON:
        random.seed(config.PRE.SEED.VALUE)

    # mixture maker
    randomMixtureMaker = RandomMixtureMaker(
        fs=config.PRE.FS,
        rooms=config.PRE.MIXTURES.RANDOM.ROOMS,
        target_angles=range(
            config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.MIN,
            config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.MAX + 1,
            config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.STEP,
        ),
        target_snrs=range(
            config.PRE.MIXTURES.RANDOM.TARGET.SNR.MIN,
            config.PRE.MIXTURES.RANDOM.TARGET.SNR.MAX + 1,
        ),
        directional_noise_numbers=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER.MAX + 1,
        ),
        directional_noise_types=config.PRE.MIXTURES.RANDOM.SOURCES.TYPES,
        directional_noise_angles=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.MAX + 1,
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.STEP,
        ),
        directional_noise_snrs=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.SNR.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.SNR.MAX + 1,
        ),
        diffuse_noise_on=config.PRE.MIXTURES.DIFFUSE.ON,
        diffuse_noise_color=config.PRE.MIXTURES.DIFFUSE.COLOR,
        diffuse_noise_ltas_eq=config.PRE.MIXTURES.DIFFUSE.LTASEQ,
        mixture_pad=config.PRE.MIXTURES.PADDING,
        mixture_rb=config.PRE.MIXTURES.REFLECTIONBOUNDARY,
        mixture_rms_jitter=range(
            config.PRE.MIXTURES.RANDOM.RMSDB.MIN,
            config.PRE.MIXTURES.RANDOM.RMSDB.MAX + 1,
        ),
        path_surrey=config.PRE.MIXTURES.PATH.SURREY,
        path_timit=config.PRE.MIXTURES.PATH.TIMIT,
        path_dcase=config.PRE.MIXTURES.PATH.DCASE,
        filelims_directional_noise=config.PRE.MIXTURES.FILELIMITS.NOISE,
        filelims_target=config.PRE.MIXTURES.FILELIMITS.TARGET,
        decay_on=config.PRE.MIXTURES.DECAY.ON,
        decay_color=config.PRE.MIXTURES.DECAY.COLOR,
        decay_rt60s=np.arange(
            config.PRE.MIXTURES.RANDOM.DECAY.RT60.MIN,
            config.PRE.MIXTURES.RANDOM.DECAY.RT60.MAX,
            config.PRE.MIXTURES.RANDOM.DECAY.RT60.STEP,
            dtype=float,
        ),
        decay_drrs=np.arange(
            config.PRE.MIXTURES.RANDOM.DECAY.DRR.MIN,
            config.PRE.MIXTURES.RANDOM.DECAY.DRR.MAX,
            config.PRE.MIXTURES.RANDOM.DECAY.DRR.STEP,
            dtype=float,
        ),
        decay_delays=np.arange(
            config.PRE.MIXTURES.RANDOM.DECAY.DELAY.MIN,
            config.PRE.MIXTURES.RANDOM.DECAY.DELAY.MAX,
            config.PRE.MIXTURES.RANDOM.DECAY.DELAY.STEP,
            dtype=float,
        ),
    )

    # scaler
    scaler = UnitRMSScaler(
        active=config.PRE.MIXTURES.SCALERMS,
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
        features=sorted(config.PRE.FEATURES),
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
    n_examples = min(10, config.PRE.MIXTURES.NUMBER)
    examples = []
    examples_index = random.sample(range(config.PRE.MIXTURES.NUMBER),
                                   n_examples)
    examples_index.sort()
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
        if i in examples_index:
            examples.append(mixture.flatten())

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
        metadata['dataset_indices'] = (i_start, i_end)
        metadatas.append(metadata)
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
        if config.PRE.MIXTURES.SAVE:
            f.create_dataset('mixtures', data=mixtures,
                             dtype=h5py.vlen_dtype(float))
            f.create_dataset('foregrounds', data=foregrounds,
                             dtype=h5py.vlen_dtype(float))
            f.create_dataset('backgrounds', data=backgrounds,
                             dtype=h5py.vlen_dtype(float))

    # save mixtures metadata
    metadatas_output_path = os.path.join(dataset_dir, 'mixture_info.json')
    with open(metadatas_output_path, 'w') as f:
        json.dump(metadatas, f)

    # save pipes
    pipes_output_path = os.path.join(dataset_dir, 'pipes.pkl')
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

    # save up to 10 random mixtures for verification
    examples_dir = os.path.join(dataset_dir, 'examples')
    if not os.path.exists(examples_dir):
        os.mkdir(examples_dir)
    else:
        for filename in os.listdir(examples_dir):
            os.remove(os.path.join(examples_dir, filename))
    for example, i in zip(examples, examples_index):
        gain = 1/example.max()
        filepath = os.path.join(examples_dir, f'example_{i}.wav')
        sf.write(filepath, gain*example.reshape(-1, 2), config.PRE.FS)

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

        peek_output_path = os.path.join(dataset_dir, filename)
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
    parser = argparse.ArgumentParser(description='create a dataset')
    parser.add_argument('input',
                        help='input dataset directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite if already exists')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
    )

    for dataset_dir in glob(args.input):
        main(dataset_dir, args.force)
