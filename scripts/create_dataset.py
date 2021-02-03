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
            config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.MAX
            + config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.STEP,
            config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.STEP,
        ),
        target_snrs=range(
            config.PRE.MIXTURES.RANDOM.TARGET.SNR.MIN,
            config.PRE.MIXTURES.RANDOM.TARGET.SNR.MAX + 1,
        ),
        dir_noise_nums=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER.MAX + 1,
        ),
        dir_noise_types=config.PRE.MIXTURES.RANDOM.SOURCES.TYPES,
        dir_noise_angles=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.MAX
            + config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.STEP,
            config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.STEP,
        ),
        dir_noise_snrs=range(
            config.PRE.MIXTURES.RANDOM.SOURCES.SNR.MIN,
            config.PRE.MIXTURES.RANDOM.SOURCES.SNR.MAX + 1,
        ),
        diffuse_noise_on=config.PRE.MIXTURES.DIFFUSE.ON,
        diffuse_noise_color=config.PRE.MIXTURES.DIFFUSE.COLOR,
        diffuse_noise_ltas_eq=config.PRE.MIXTURES.DIFFUSE.LTASEQ,
        mixture_pad=config.PRE.MIXTURES.PADDING,
        mixture_rb=config.PRE.MIXTURES.REFLECTIONBOUNDARY,
        mixture_rms_jitter_on=config.PRE.MIXTURES.RANDOM.RMSDB.ON,
        mixture_rms_jitters=range(
            config.PRE.MIXTURES.RANDOM.RMSDB.MIN,
            config.PRE.MIXTURES.RANDOM.RMSDB.MAX + 1,
        ),
        path_surrey=config.PRE.MIXTURES.PATH.SURREY,
        path_target=config.PRE.MIXTURES.PATH.TARGET,
        path_dcase=config.PRE.MIXTURES.PATH.DCASE,
        filelims_dir_noise=config.PRE.MIXTURES.FILELIMITS.NOISE,
        filelims_target=config.PRE.MIXTURES.FILELIMITS.TARGET,
        decay_on=config.PRE.MIXTURES.DECAY.ON,
        decay_color=config.PRE.MIXTURES.DECAY.COLOR,
        decay_rt60s=np.arange(
            config.PRE.MIXTURES.RANDOM.DECAY.RT60.MIN,
            config.PRE.MIXTURES.RANDOM.DECAY.RT60.MAX
            + config.PRE.MIXTURES.RANDOM.DECAY.RT60.STEP,
            config.PRE.MIXTURES.RANDOM.DECAY.RT60.STEP,
            dtype=float,
        ),
        decay_drrs=np.arange(
            config.PRE.MIXTURES.RANDOM.DECAY.DRR.MIN,
            config.PRE.MIXTURES.RANDOM.DECAY.DRR.MAX
            + config.PRE.MIXTURES.RANDOM.DECAY.DRR.STEP,
            config.PRE.MIXTURES.RANDOM.DECAY.DRR.STEP,
            dtype=float,
        ),
        decay_delays=np.arange(
            config.PRE.MIXTURES.RANDOM.DECAY.DELAY.MIN,
            config.PRE.MIXTURES.RANDOM.DECAY.DELAY.MAX
            + config.PRE.MIXTURES.RANDOM.DECAY.DELAY.STEP,
            config.PRE.MIXTURES.RANDOM.DECAY.DELAY.STEP,
            dtype=float,
        ),
        seed_on=config.PRE.SEED.ON,
        seed_value=config.PRE.SEED.VALUE,
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
        labels=sorted(config.PRE.LABELS),
    )

    # main loop intialization
    features = []
    labels = []
    mixtures = []
    foregrounds = []
    backgrounds = []
    noises = []
    reverbs = []
    metadatas = []
    i_start = 0
    total_time = 0
    start_time = time.time()
    n_examples = min(10, config.PRE.MIXTURES.NUMBER)
    examples = []
    examples_index = random.sample(range(config.PRE.MIXTURES.NUMBER),
                                   n_examples)

    # SEED AGAIN FOR REPRODUCIBILITY BECAUSE RANDOMIZING THE EXAMPLE INDEXES
    # LEADS TO DIFFERENT SEEDS COMING INTO THE RANDOMMIXTUREMAKER OBJECT FOR
    # TWO IDENTICAL DATASETS WITH DIFFERENT NUMBER OF MIXTURES!!!
    #
    # This was probably fixed after having implemented the independent random
    # generators for each random parameter, but I am leaving it for safety
    if config.PRE.SEED.ON:
        random.seed(config.PRE.SEED.VALUE)

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
        mixtureObject, metadata = randomMixtureMaker.make()
        if config.PRE.MIXTURES.SAVE:
            mixtures.append(mixtureObject.mixture.flatten())
            foregrounds.append(mixtureObject.foreground.flatten())
            backgrounds.append(mixtureObject.background.flatten())
            noises.append(mixtureObject.noise.flatten())
            reverbs.append(mixtureObject.late_target.flatten())
        if i in examples_index:
            examples.append(mixtureObject.mixture.flatten())

        # scale signal
        scaler.fit(mixtureObject.mixture)
        mixtureObject.transform(scaler.scale)
        scaler.__init__(scaler.active)

        # apply filterbank
        mixtureObject.transform(filterbank.filt)

        # frame
        mixtureObject.transform(framer.frame)

        # extract features
        features.append(featureExtractor.run(mixtureObject.mixture))

        # extract labels
        labels.append(labelExtractor.run(mixtureObject))

        # save indices
        i_end = i_start + len(mixtureObject)
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
            f.create_dataset(
                'mixtures',
                data=np.array(mixtures, dtype=object),
                dtype=h5py.vlen_dtype(float),
            )
            f.create_dataset(
                'foregrounds',
                data=np.array(foregrounds, dtype=object),
                dtype=h5py.vlen_dtype(float),
            )
            f.create_dataset(
                'backgrounds',
                data=np.array(backgrounds, dtype=object),
                dtype=h5py.vlen_dtype(float),
                )
            f.create_dataset(
                'noises',
                data=np.array(noises, dtype=object),
                dtype=h5py.vlen_dtype(float),
            )
            f.create_dataset(
                'reverbs',
                data=np.array(reverbs, dtype=object),
                dtype=h5py.vlen_dtype(float),
            )

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
        yticks_major = [index[0] for index in labelExtractor.indices]
        yticks_major.append(labelExtractor.indices[-1][1])
        axes[1].set_yticks(yticks_major)
        axes[1].set_yticklabels(yticks_major)
        yticks_minor = np.mean(labelExtractor.indices, axis=1)
        axes[1].set_yticks(yticks_minor, minor=True)
        axes[1].set_yticklabels(labelExtractor.labels, minor=True)
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
    parser.add_argument('input', nargs='+',
                        help='input dataset directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite if already exists')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
    )

    dataset_dirs = []
    for input_ in args.input:
        if not glob(input_):
            logging.info(f'Dataset not found: {input_}')
        dataset_dirs += glob(input_)
    for dataset_dir in dataset_dirs:
        main(dataset_dir, args.force)
