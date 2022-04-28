import argparse
import json
import logging
import os
import random
import shutil
import tarfile
import tempfile
import time

import soundfile as sf

from brever.config import get_config
from brever.logger import set_logger
from brever.mixture import RandomMixtureMaker


def main():
    # check if already created
    mix_info_path = os.path.join(args.input, 'mixture_info.json')
    if os.path.exists(mix_info_path) and not args.force:
        raise FileExistsError(f'dataset already created: {mix_info_path}')

    # load config file
    config_path = os.path.join(args.input, 'config.yaml')
    config = get_config(config_path)

    # init logger
    log_file = os.path.join(args.input, 'log.log')
    set_logger(log_file)
    logging.info(f'Creating {args.input}')
    logging.info(config.to_dict())

    # output directory or archive
    mix_dirname = 'audio'
    if args.no_tar:
        mix_dirpath = os.path.join(args.input, mix_dirname)
        if os.path.exists(mix_dirpath):
            shutil.rmtree(mix_dirpath)
        os.mkdir(mix_dirpath)
    else:
        archive_path = os.path.join(args.input, f'{mix_dirname}.tar')
        archive = tarfile.open(archive_path, 'w')

    # seed for reproducibility
    random.seed(config.SEED)

    # mixture maker
    randomMixtureMaker = RandomMixtureMaker(
        fs=config.FS,
        seed=config.SEED,
        padding=config.PADDING,
        uniform_tmr=config.UNIFORM_TMR,
        reflection_boundary=config.REFLECTION_BOUNDARY,
        speakers=config.SPEAKERS,
        noises=config.NOISES,
        rooms=config.ROOMS,
        snr_dist_name=config.TARGET.SNR.DIST_NAME,
        snr_dist_args=config.TARGET.SNR.DIST_ARGS,
        target_angle=config.TARGET.ANGLE,
        noise_num=config.NOISE.NUMBER,
        noise_angle=config.NOISE.ANGLE,
        ndr_dist_name=config.NOISE.NDR.DIST_NAME,
        ndr_dist_args=config.NOISE.NDR.DIST_ARGS,
        diffuse=config.DIFFUSE.TOGGLE,
        diffuse_color=config.DIFFUSE.COLOR,
        diffuse_ltas_eq=config.DIFFUSE.LTAS_EQ,
        decay=config.DECAY.TOGGLE,
        decay_color=config.DECAY.COLOR,
        rt60_dist_name=config.DECAY.RT60.DIST_NAME,
        rt60_dist_args=config.DECAY.RT60.DIST_ARGS,
        drr_dist_name=config.DECAY.DRR.DIST_NAME,
        drr_dist_args=config.DECAY.DRR.DIST_ARGS,
        delay_dist_name=config.DECAY.DELAY.DIST_NAME,
        delay_dist_args=config.DECAY.DELAY.DIST_ARGS,
        rms_jitter_dist_name=config.RMS_JITTER.DIST_NAME,
        rms_jitter_dist_args=config.RMS_JITTER.DIST_ARGS,
        speech_files=config.FILES.SPEECH,
        noise_files=config.FILES.NOISE,
        room_files=config.FILES.ROOM,
    )

    # main loop intialization
    metadatas = []
    time_spent = 0
    start_time = time.time()
    duration = 0
    i = 0

    # main loop
    while duration < config.DURATION:

        # estimate time remaining and show progress
        if i == 0:
            logging.info(f'Making mixture {i+1}...')
        else:
            time_per_mix = time_spent/i
            avg_mix_duration = duration/i
            estimated_n_mix = config.DURATION/avg_mix_duration
            etr = (estimated_n_mix-i)*time_per_mix
            h, m, s = int(etr/3600), int(etr % 3600/60), int(etr % 60)
            logging.info(f'Making mixture {i+1}... '
                         f'ETA: {h} h {m} m {s} s, '
                         f'/mix: {time_per_mix:.2f} s')

        # make mixture and save
        mixObject, metadata = randomMixtureMaker.make()
        for name in config.COMPONENTS:
            filename = f'{i:05d}_{name}.flac'
            if args.no_tar:
                filepath = os.path.join(mix_dirpath, filename)
                sf.write(filepath, getattr(mixObject, name), config.FS)
            else:
                temp = tempfile.NamedTemporaryFile(
                    prefix='brever_',
                    suffix='.flac',
                    delete=False,
                )
                sf.write(temp, getattr(mixObject, name), config.FS)
                temp.close()
                arcname = os.path.join(mix_dirname, filename)
                archive.add(temp.name, arcname=arcname)
                os.remove(temp.name)
        metadatas.append(metadata)

        # update duration and time spent
        duration += len(mixObject)/config.FS
        time_spent = time.time() - start_time
        i += 1

    # save mixtures metadata
    with open(mix_info_path, 'w') as f:
        json.dump(metadatas, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create a dataset')
    parser.add_argument('input',
                        help='dataset directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite if already exists')
    parser.add_argument('--no-tar', action='store_true',
                        help='do not save mixtures in tar archive')
    args = parser.parse_args()
    main()
