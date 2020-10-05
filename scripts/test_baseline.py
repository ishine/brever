import argparse
import logging
from glob import glob
import os
import pickle
import time
import sys

import h5py
import matlab
import matlab.engine
import numpy as np


def main(dataset_dir, force, eng):
    logging.info(f'Processing {dataset_dir}')
    dataset_path = os.path.join(dataset_dir, 'dataset.hdf5')

    # load pipes
    logging.info('Loading pipes...')
    pipes_file = os.path.join(dataset_dir, 'pipes.pkl')
    with open(pipes_file, 'rb') as f:
        pipes = pickle.load(f)
    scaler = pipes['scaler']
    filterbank = pipes['filterbank']
    randomMixtureMaker = pipes['randomMixtureMaker']
    fs = randomMixtureMaker.fs

    # main loop
    total_time = 0
    start_time = time.time()
    dpesqs = []
    with h5py.File(dataset_path, 'r') as f:
        n = len(f['mixtures'])
        for k in range(n):
            if k == 0:
                logging.info(f'Calculating PESQ for mixture {k}/{n}...')
            else:
                etr = (n-k)*total_time/k
                logging.info((f'Calculating PESQ for mixture {k}/{n}... '
                              f'ETR: {int(etr/60)} m {int(etr%60)} s'))
            mixture = f['mixtures'][k].reshape(-1, 2)
            foreground = f['foregrounds'][k].reshape(-1, 2)

            # scale signal
            scaler.fit(mixture)
            mixture = scaler.scale(mixture)
            foreground = scaler.scale(foreground)
            scaler.__init__(scaler.active)

            # apply filterbank
            mixture_filt = filterbank.filt(mixture)
            foreground_filt = filterbank.filt(foreground)

            # apply reverse filterbank
            mixture_ref = filterbank.rfilt(mixture_filt)
            foreground_ref = filterbank.rfilt(foreground_filt)

            # convert to matlab float
            mixture_ref = matlab.single(mixture_ref.tolist())

            # call baseline model
            P = eng.configSTFT(
                float(fs),
                32e-3,
                0.5,
                'hann',
                'wola',
            )
            mixture_enhanced = eng.noiseReductionDoerbecker(
                mixture_ref,
                float(fs),
                P,
            )

            # convert back to numpy array
            mixture_ref = np.array(mixture_ref)
            mixture_enhanced = np.array(mixture_enhanced)

            # remove noise-only parts
            npad = round(randomMixtureMaker.padding*fs)
            mixture_enhanced = mixture_enhanced[npad:-npad]
            mixture_ref = mixture_ref[npad:-npad]
            foreground_ref = foreground_ref[npad:-npad]

            # flatten and convert to matlab float
            mixture_enhanced = matlab.single(
                mixture_enhanced.sum(axis=1, keepdims=True).tolist())
            mixture_ref = matlab.single(
                mixture_ref.sum(axis=1, keepdims=True).tolist())
            foreground_ref = matlab.single(
                foreground_ref.sum(axis=1, keepdims=True).tolist())

            # calculate PESQ
            pesq_before = eng.pesq(foreground_ref, mixture_ref,
                                   randomMixtureMaker.fs)
            pesq_after = eng.pesq(foreground_ref, mixture_enhanced,
                                  randomMixtureMaker.fs)
            dpesq = pesq_after - pesq_before
            dpesqs.append(dpesq)
            logging.info(f'Delta PESQ: {dpesq:.2f}')

            # update time spent
            total_time = time.time() - start_time

    PESQ = np.mean(dpesqs)
    output_dir = 'matlab/scores'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    basename = os.path.basename(dataset_dir)
    output_path = os.path.join(output_dir, basename)
    np.save(output_path, PESQ)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test baseline system')
    parser.add_argument('input',
                        help='input dataset directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    if glob(args.input):

        # load matlab engine
        logging.info('Starting MATLAB engine...')
        eng = matlab.engine.start_matlab()
        paths = [
            'matlab',
            'matlab/loizou',
            'matlab/stft-framework/stft-framework/src/tools',
        ]
        for path in paths:
            eng.addpath(path, nargout=0)

        for dataset_dir in glob(args.input):
            main(dataset_dir, args.force, eng)
