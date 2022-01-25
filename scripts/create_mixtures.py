import argparse
# from functools import partial
from glob import glob
import logging
import os
import pprint
import random
import shutil
import sys
import time

import numpy as np
import soundfile as sf

from brever import RandomMixtureMaker
from brever.config import defaults
import brever.management as bm
from brever.utils import UnitRMSScaler


def add_to_dset(dset, data):
    if dset.shape[0] == 0:
        dset.resize(data.shape)
    else:
        dset.resize(dset.shape[0]+len(data), axis=0)
    dset[-len(data):] = data


def main(dataset_dir, force):
    # check if dataset already exists
    metadatas_output_path = os.path.join(dataset_dir, 'mixture_info.json')
    if os.path.exists(metadatas_output_path) and not force:
        logging.info('Dataset already exists')
        return

    # load config file
    config = defaults()
    config_file = os.path.join(dataset_dir, 'config.yaml')
    config.update(bm.read_yaml(config_file))

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
    randomMixtureMaker = RandomMixtureMaker(
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
        filelims_room=config.PRE.MIX.FILELIMITS.ROOM,
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
    scaler = UnitRMSScaler(
        active=config.PRE.MIX.SCALERMS,
    )

    # component names
    component_names = [
        'mixture',
        'foreground',
        'background',
        # 'noise',
        # 'late_target',
    ]

    # output directory
    mixtures_dir = os.path.join(dataset_dir, 'mixtures')
    if os.path.exists(mixtures_dir):
        shutil.rmtree(mixtures_dir)
    os.mkdir(mixtures_dir)

    # main loop intialization
    metadatas = []
    total_time = 0
    start_time = time.time()
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
        for name in component_names:
            filename = f'{i:05d}_{name}.wav'
            filepath = os.path.join(mixtures_dir, filename)
            sf.write(filepath, getattr(mixObject, name), config.PRE.FS)

        # update total duration
        total_duration += len(mixObject)/config.PRE.FS

        # scale signal
        scaler.fit(mixObject.mixture)
        mixObject.transform(scaler.scale)
        if scaler.active:
            metadata['rms_scaler_gain'] = scaler.gain

        # save metadata
        metadatas.append(metadata)

        # update time spent
        total_time = time.time() - start_time

        i += 1

    # save mixtures metadata
    bm.dump_json(metadatas, metadatas_output_path)

    # write full config file
    full_config_file = os.path.join(dataset_dir, 'config_full.yaml')
    bm.dump_yaml(config.to_dict(), full_config_file)


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
