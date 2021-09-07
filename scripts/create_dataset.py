import os
import argparse
import logging
import time
import pprint
import pickle
import random
from glob import glob
import sys
import copy
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import soundfile as sf

from brever.config import defaults
import brever.classes as bpipes
from brever.utils import wola
import brever.modelmanagement as bmm


def add_to_vlen_dset(dset, data):
    dset.resize(dset.shape[0]+1, axis=0)
    dset[-1] = data.flatten()


def add_to_dset(dset, data):
    if dset.shape[0] == 0:
        dset.resize(data.shape)
    else:
        dset.resize(dset.shape[0]+len(data), axis=0)
    dset[-len(data):] = data


def main(dataset_dir, force):
    # check if dataset already exists
    datasets_output_path = os.path.join(dataset_dir, 'dataset.hdf5')
    metadatas_output_path = os.path.join(dataset_dir, 'mixture_info.json')
    if os.path.exists(metadatas_output_path) and not force:
        logging.info('Dataset already exists')
        return

    # load config file
    config = defaults()
    config_file = os.path.join(dataset_dir, 'config.yaml')
    config.update(bmm.read_yaml(config_file))

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
    logging.info(f'Processing {dataset_dir}...')
    logging.info(pprint.pformat({'PRE': config.PRE.to_dict()}))

    # seed for reproducibility
    if config.PRE.SEED.ON:
        random.seed(config.PRE.SEED.VALUE)

    # mixture maker
    randomMixtureMaker = bpipes.RandomMixtureMaker(
        fs=config.PRE.FS,
        rooms=config.PRE.MIX.RANDOM.ROOMS,
        speakers=config.PRE.MIX.RANDOM.TARGET.SPEAKERS,
        target_snr_dist_name=config.PRE.MIX.RANDOM.TARGET.SNR.DISTNAME,
        target_snr_dist_args=config.PRE.MIX.RANDOM.TARGET.SNR.DISTARGS,
        target_angle_min=config.PRE.MIX.RANDOM.TARGET.ANGLE.MIN,
        target_angle_max=config.PRE.MIX.RANDOM.TARGET.ANGLE.MAX,
        dir_noise_nums=range(
            config.PRE.MIX.RANDOM.SOURCES.NUMBER.MIN,
            config.PRE.MIX.RANDOM.SOURCES.NUMBER.MAX + 1,
        ),
        dir_noise_types=config.PRE.MIX.RANDOM.SOURCES.TYPES,
        dir_noise_snrs=range(
            config.PRE.MIX.RANDOM.SOURCES.SNR.MIN,
            config.PRE.MIX.RANDOM.SOURCES.SNR.MAX + 1,
        ),
        dir_noise_angle_min=config.PRE.MIX.RANDOM.SOURCES.ANGLE.MIN,
        dir_noise_angle_max=config.PRE.MIX.RANDOM.SOURCES.ANGLE.MAX,
        diffuse_noise_on=config.PRE.MIX.DIFFUSE.ON,
        diffuse_noise_color=config.PRE.MIX.DIFFUSE.COLOR,
        diffuse_noise_ltas_eq=config.PRE.MIX.DIFFUSE.LTASEQ,
        mixture_pad=config.PRE.MIX.PADDING,
        mixture_rb=config.PRE.MIX.REFLECTIONBOUNDARY,
        mixture_rms_jitter_on=config.PRE.MIX.RANDOM.RMSDB.ON,
        mixture_rms_jitters=range(
            config.PRE.MIX.RANDOM.RMSDB.MIN,
            config.PRE.MIX.RANDOM.RMSDB.MAX + 1,
        ),
        filelims_dir_noise=config.PRE.MIX.FILELIMITS.NOISE,
        filelims_target=config.PRE.MIX.FILELIMITS.TARGET,
        decay_on=config.PRE.MIX.DECAY.ON,
        decay_color=config.PRE.MIX.DECAY.COLOR,
        decay_rt60s=np.arange(
            config.PRE.MIX.RANDOM.DECAY.RT60.MIN,
            config.PRE.MIX.RANDOM.DECAY.RT60.MAX
            + config.PRE.MIX.RANDOM.DECAY.RT60.STEP,
            config.PRE.MIX.RANDOM.DECAY.RT60.STEP,
            dtype=float,
        ),
        decay_drr_dist_name=config.PRE.MIX.RANDOM.DECAY.DRR.DISTNAME,
        decay_drr_dist_args=config.PRE.MIX.RANDOM.DECAY.DRR.DISTARGS,
        decay_delays=np.arange(
            config.PRE.MIX.RANDOM.DECAY.DELAY.MIN,
            config.PRE.MIX.RANDOM.DECAY.DELAY.MAX
            + config.PRE.MIX.RANDOM.DECAY.DELAY.STEP,
            config.PRE.MIX.RANDOM.DECAY.DELAY.STEP,
            dtype=float,
        ),
        seed_on=config.PRE.SEED.ON,
        seed_value=config.PRE.SEED.VALUE,
        uniform_tmr=config.PRE.MIX.RANDOM.UNIFORMTMR,
    )

    # scaler
    scaler = bpipes.UnitRMSScaler(
        active=config.PRE.MIX.SCALERMS,
    )

    # filterbank
    filterbank = bpipes.MultiThreadFilterbank(
        kind=config.PRE.FILTERBANK.KIND,
        n_filters=config.PRE.FILTERBANK.NFILTERS,
        f_min=config.PRE.FILTERBANK.FMIN,
        f_max=config.PRE.FILTERBANK.FMAX,
        fs=config.PRE.FS,
        order=config.PRE.FILTERBANK.ORDER,
    )

    # framer
    framer = bpipes.Framer(
        frame_length=config.PRE.FRAMER.FRAMELENGTH,
        hop_length=config.PRE.FRAMER.HOPLENGTH,
        window=config.PRE.FRAMER.WINDOW,
        center=config.PRE.FRAMER.CENTER,
    )

    # feature extractor
    featureExtractor = bpipes.FeatureExtractor(
        features=sorted(config.PRE.FEATURES),
    )

    # label extractor
    labelExtractor = bpipes.LabelExtractor(
        labels=sorted(config.PRE.LABELS),
    )

    # initialize hdf5 datasets
    h5f = h5py.File(datasets_output_path, 'w')
    dset_feats = h5f.create_dataset(
        'features',
        (0, 0),
        maxshape=(None, None),
        dtype='f4',
    )
    dset_labels = h5f.create_dataset(
        'labels',
        (0, 0),
        maxshape=(None, None),
        dtype='f4',
    )
    dset_indexes = h5f.create_dataset(
        'indexes',
        (0,),
        maxshape=(None,),
        dtype='i4',
    )
    if config.PRE.MIX.SAVE:
        vlen_dsets = {}
        component_names = [
            'mixture',
            'foreground',
            'background',
            'noise',
            'late_target',
        ]
        for name in component_names:
            vlen_dsets[name] = h5f.create_dataset(
                name,
                (0,),
                dtype=h5py.vlen_dtype('f4'),
                maxshape=(None,),
            )
            vlen_dsets[f'{name}_ref'] = h5f.create_dataset(
                f'{name}_ref',
                (0,),
                dtype=h5py.vlen_dtype('f4'),
                maxshape=(None,),
            )
            for label in labelExtractor.labels:
                vlen_dsets[f'{name}_oracle_{label}'] = h5f.create_dataset(
                    f'{name}_oracle_{label}',
                    (0,),
                    dtype=h5py.vlen_dtype('f4'),
                    maxshape=(None,),
                )

    # main loop intialization
    metadatas = []
    i_start = 0
    total_time = 0
    start_time = time.time()
    examples = []
    examples_index = list(range(10))
    total_duration = 0
    i = 0

    # main loop
    while total_duration < config.PRE.MIX.TOTALDURATION:

        # estimate time remaining and show progress
        if i == 0:
            logging.info(f'Processing mixture {i+1}...')
        else:
            time_per_mix = total_time/i
            mix_avg_dur = total_duration/i
            n_mix_esti = config.PRE.MIX.TOTALDURATION/mix_avg_dur
            etr = (n_mix_esti-i)*time_per_mix
            h, m, s = int(etr/3600), int(etr % 3600/60), int(etr % 60)
            logging.info(f'Processing mixture {i+1}... '
                         f'ETR: {h} h {m} m {s} s, '
                         f'/mix: {time_per_mix:.2f} s')

        # make mixture and save
        mixObject, metadata = randomMixtureMaker.make()
        if config.PRE.MIX.SAVE:
            for name in component_names:
                add_to_vlen_dset(
                    vlen_dsets[name],
                    getattr(mixObject, name).flatten(),
                )
        if i in examples_index:
            examples.append(mixObject.mixture.flatten())

        # update total duration
        total_duration += len(mixObject)/config.PRE.FS

        # scale signal
        scaler.fit(mixObject.mixture)
        mixObject.transform(scaler.scale)
        metadata['rms_scaler_gain'] = scaler.gain
        scaler.__init__(scaler.active)

        # apply filterbank
        mixObject.transform(filterbank.filt)

        # reverse filter to obtain reference signals and save
        if config.PRE.MIX.SAVE:
            mixRef = copy.deepcopy(mixObject)
            mixRef.transform(filterbank.rfilt)
            for name in component_names:
                add_to_vlen_dset(
                    vlen_dsets[f'{name}_ref'],
                    getattr(mixRef, name).flatten(),
                )
            del mixRef

        # keep a copy of the mixture object for later
        if config.PRE.MIX.SAVE:
            mixCopy = copy.deepcopy(mixObject)

        # frame
        mixObject.transform(framer.frame)

        # extract features
        features = featureExtractor.run(mixObject.mixture)

        # extract labels
        labels = labelExtractor.run(mixObject)

        # apply label and reverse filter to obtain oracle signals
        if config.PRE.MIX.SAVE:
            for (j_start, j_end), label in zip(labelExtractor.indices,
                                               labelExtractor.labels):
                mixOracle = copy.deepcopy(mixCopy)
                mask = labels[:, j_start:j_end]
                mask = wola(mask, trim=len(mixOracle))
                mask = mask[:, :, np.newaxis]
                mixOracle.transform(partial(np.multiply, mask))
                mixOracle.transform(filterbank.rfilt)
                for name in component_names:
                    add_to_vlen_dset(
                        vlen_dsets[f'{name}_oracle_{label}'],
                        getattr(mixOracle, name).flatten(),
                    )
            del mixOracle
            del mixCopy

        # create indexes array
        indexes = np.full(len(features), i, dtype=int)

        # save features and labels
        for dset, data in zip(
                    [dset_feats, dset_labels, dset_indexes],
                    [features, labels, indexes],
                ):
            add_to_dset(dset, data)

        # save indices
        i_end = i_start + len(mixObject)
        metadata['dataset_indices'] = (i_start, i_end)
        i_start = i_end

        # save metadata
        metadatas.append(metadata)

        # update time spent
        total_time = time.time() - start_time

        i += 1

    # close hdf5 file
    h5f.close()

    # save mixtures metadata
    bmm.dump_json(metadatas, metadatas_output_path)

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
        plt.close(fig)

    if config.PRE.PLOT.ON:
        x = features[:config.PRE.PLOT.NSAMPLES]
        y = labels[:config.PRE.PLOT.NSAMPLES]
        plot(x, y, 'peek.png')

        standardizer = bpipes.Standardizer()
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
