import json
import logging
import os
import random
import shutil
import time

import soundfile as sf

from brever.logger import set_logger
from brever.mixture import RandomMixtureMaker


def main():
    dset_dir = 'tests/test_dataset'
    if not os.path.exists(dset_dir):
        os.mkdir(dset_dir)

    # init logger
    log_file = os.path.join(dset_dir, 'log.log')
    set_logger(log_file)
    logging.info(f'Creating {dset_dir}')

    # output directory
    mixtures_dir = os.path.join(dset_dir, 'audio')
    if os.path.exists(mixtures_dir):
        shutil.rmtree(mixtures_dir)
    os.mkdir(mixtures_dir)

    # seed for reproducibility
    random.seed(0)

    # mixture maker
    randomMixtureMaker = RandomMixtureMaker(seed=0)

    # main loop intialization
    metadatas = []
    time_spent = 0
    start_time = time.time()
    duration = 0
    i = 0
    total_duration = 30
    components = ['mixture', 'foreground', 'background']
    fs = 16000

    # main loop
    while duration < total_duration:

        # estimate time remaining and show progress
        if i == 0:
            logging.info(f'Making mixture {i+1}...')
        else:
            time_per_mix = time_spent/i
            avg_mix_duration = duration/i
            estimated_n_mix = total_duration/avg_mix_duration
            etr = (estimated_n_mix-i)*time_per_mix
            h, m, s = int(etr/3600), int(etr % 3600/60), int(etr % 60)
            logging.info(f'Making mixture {i+1}... '
                         f'ETA: {h} h {m} m {s} s, '
                         f'/mix: {time_per_mix:.2f} s')

        # make mixture and save
        mixObject, metadata = randomMixtureMaker.make()
        for name in components:
            filename = f'{i:05d}_{name}.wav'
            filepath = os.path.join(mixtures_dir, filename)
            sf.write(filepath, getattr(mixObject, name), fs)
        metadatas.append(metadata)

        # update duration and time spent
        duration += len(mixObject)/fs
        time_spent = time.time() - start_time
        i += 1

    # save mixtures metadata
    mix_info_path = os.path.join(dset_dir, 'mixture_info.json')
    with open(mix_info_path, 'w') as f:
        json.dump(metadatas, f)


if __name__ == '__main__':
    main()
