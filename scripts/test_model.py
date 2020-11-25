import os
import argparse
from glob import glob
import pickle
import logging
import sys

import yaml
import numba  # noqa: F401
import torch
import numpy as np
import h5py
import soundfile as sf

from brever.config import defaults
from brever.utils import wola
from brever.pytorchtools import (Feedforward, H5Dataset, TensorStandardizer,
                                 evaluate, get_files_mean_and_std,
                                 StateTensorStandardizer)


def main(model_dir, force):
    logging.info(f'Processing {model_dir}')

    # check if model is already tested
    output_pesq_path = os.path.join(model_dir, 'pesq_scores.mat')
    output_mse_path = os.path.join(model_dir, 'mse_scores.npy')
    if os.path.exists(output_pesq_path) and os.path.exists(output_mse_path):
        if not force:
            logging.info('Model is already tested!')
            return

    # check if model is trained
    train_loss_path = os.path.join(model_dir, 'train_losses.npy')
    val_loss_path = os.path.join(model_dir, 'val_losses.npy')
    if os.path.exists(train_loss_path) and os.path.exists(val_loss_path):
        pass
    else:
        logging.info('Model is not trained!')
        return

    # load config file
    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    # seed for reproducibility
    torch.manual_seed(0)

    # get mean and std
    if config.POST.STANDARDIZATION.FILEBASED:
        pass
    else:
        stat_path = os.path.join(model_dir, 'statistics.npy')
        logging.info('Loading mean and std...')
        mean, std = np.load(stat_path)

    # initialize and load network
    logging.info('Loading model...')
    model_args_path = os.path.join(model_dir, 'model_args.yaml')
    model = Feedforward.build(model_args_path)
    state_file = os.path.join(model_dir, 'checkpoint.pt')
    model.load_state_dict(torch.load(state_file))

    # initialize criterion
    criterion = getattr(torch.nn, config.MODEL.CRITERION)()

    # load pipes
    logging.info('Loading pipes...')
    pipes_file = os.path.join(config.POST.PATH.TRAIN, 'pipes.pkl')
    with open(pipes_file, 'rb') as f:
        pipes = pickle.load(f)
    scaler = pipes['scaler']
    filterbank = pipes['filterbank']

    # main loop
    snrs = [0, 3, 6, 9, 12, 15]
    room_aliases = [
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
    ]
    MSE = np.zeros((len(snrs), len(room_aliases)))
    for i, snr in enumerate(snrs):
        for j, room_alias in enumerate(room_aliases):
            # build dataset directory name
            suffix = f'snr{snr}_room{room_alias[-1].upper()}'
            test_dataset_dir = f'{config.POST.PATH.TEST}_{suffix}'
            logging.info(f'Processing {test_dataset_dir}:')

            # initialize dataset and dataloader
            test_dataset = H5Dataset(
                dirpath=test_dataset_dir,
                features=config.POST.FEATURES,
                load=config.POST.LOAD,
                stack=config.POST.STACK,
                decimation=1,  # there must not be decimation during testing
                dct=config.POST.DCT.ON,
                n_dct=config.POST.DCT.NCOEFF,
            )
            test_dataloader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=config.MODEL.BATCHSIZE,
                shuffle=config.MODEL.SHUFFLE,
                num_workers=config.MODEL.NWORKERS,
                drop_last=True,
            )
            if config.POST.STANDARDIZATION.FILEBASED:
                test_means, test_stds = get_files_mean_and_std(
                    test_dataset,
                    config.POST.STANDARDIZATION.UNIFORMFEATURES,
                )
                test_dataset.transform = StateTensorStandardizer(
                    test_means,
                    test_stds,
                )
            else:
                test_dataset.transform = TensorStandardizer(mean, std)
            logging.info('Calculating MSE...')
            MSE[i, j] = evaluate(
                model=model,
                criterion=criterion,
                dataloader=test_dataloader,
                load=config.POST.LOAD,
                cuda=config.MODEL.CUDA,
            )

            # enhance mixtures for PESQ calculation
            with h5py.File(test_dataset.filepath, 'r') as f:
                n = len(f['mixtures'])
                for k in range(n):
                    logging.info(f'Enhancing mixture {k}/{n}...')

                    # load mixture
                    mixture = f['mixtures'][k].reshape(-1, 2)
                    foreground = f['foregrounds'][k].reshape(-1, 2)
                    i_start, i_end = test_dataset.file_indices[k]

                    # scale signal
                    scaler.fit(mixture)
                    mixture = scaler.scale(mixture)
                    foreground = scaler.scale(foreground)
                    scaler.__init__(scaler.active)

                    # apply filterbank
                    mixture_filt = filterbank.filt(mixture)
                    foreground_filt = filterbank.filt(foreground)

                    # extract features
                    features, _ = test_dataset[i_start:i_end]
                    features = torch.from_numpy(features).float()
                    if config.MODEL.CUDA:
                        features = features.cuda()

                    # make mask prediction
                    model.eval()
                    with torch.no_grad():
                        PRM = model(features)
                        if config.MODEL.CUDA:
                            PRM = PRM.cpu()
                        PRM = PRM.numpy()

                    # apply predicted mask
                    PRM_ = wola(PRM, trim=len(mixture_filt))[:, :, np.newaxis]
                    mixture_enhanced = filterbank.rfilt(mixture_filt*PRM_)
                    mixture_ref = filterbank.rfilt(mixture_filt)
                    foreground_ref = filterbank.rfilt(foreground_filt)

                    # write mixtures
                    gain = 1/mixture.max()
                    output_dir = os.path.join(model_dir, 'audio', suffix)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    sf.write(
                        os.path.join(output_dir, f'mixture_enhanced_{k}.wav'),
                        mixture_enhanced*gain,
                        config.PRE.FS,
                    )
                    sf.write(
                        os.path.join(output_dir, f'mixture_ref_{k}.wav'),
                        mixture_ref*gain,
                        config.PRE.FS,
                    )
                    sf.write(
                        os.path.join(output_dir, f'foreground_ref_{k}.wav'),
                        foreground_ref*gain,
                        config.PRE.FS,
                    )

    # save MSE
    np.save(os.path.join(model_dir, 'mse_scores.npy'), MSE)

    # calculate PESQ on matlab
    try:
        import matlab
        import matlab.engine
    except Exception:
        logging.info(('Matlab engine import failed. You will have to manually'
                      'call testModel.m to calculate PESQ scores.'))
    else:
        logging.info('Starting MATLAB engine...')
        eng = matlab.engine.start_matlab()
        eng.addpath('matlab/loizou', nargout=0)
        eng.addpath('matlab', nargout=0)
        logging.info('Calculating PESQ...')
        eng.testModel(
            model_dir,
            config.PRE.FS,
            config.PRE.MIXTURES.PADDING,
            nargout=0,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('input', nargs='+',
                        help='input model directories')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    model_dirs = []
    for input_ in args.input:
        if not glob(input_):
            logging.info(f'Model not found: {input_}')
        model_dirs += glob(input_)
    for model_dir in model_dirs:
        main(model_dir, args.force)
