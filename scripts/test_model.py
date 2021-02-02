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
from brever.utils import wola, segmental_scores
from brever.pytorchtools import (Feedforward, H5Dataset, TensorStandardizer,
                                 evaluate, get_files_mean_and_std,
                                 StateTensorStandardizer)


def main(model_dir, force, no_cuda):
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
    model.load_state_dict(torch.load(state_file, map_location='cpu'))
    if config.MODEL.CUDA and not no_cuda:
        model = model.cuda()

    # initialize criterion
    criterion = getattr(torch.nn, config.MODEL.CRITERION)()

    # main loop
    snrs = [0, 3, 6, 9, 12, 15]
    room_aliases = [
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
    ]
    MSE = np.empty((len(snrs), len(room_aliases)))
    seg_scores = np.empty((len(snrs), len(room_aliases)), dtype=object)
    seg_scores_oracle = np.empty((len(snrs), len(room_aliases)), dtype=object)
    for i, snr in enumerate(snrs):
        for j, room_alias in enumerate(room_aliases):
            # build dataset directory name
            suffix = f'snr{snr}_room{room_alias[-1].upper()}'
            test_dataset_dir = f'{config.POST.PATH.TEST}_{suffix}'
            logging.info(f'Processing {test_dataset_dir}:')

            # load pipes
            logging.info('Loading pipes...')
            pipes_file = os.path.join(test_dataset_dir, 'pipes.pkl')
            with open(pipes_file, 'rb') as f:
                pipes = pickle.load(f)
            scaler = pipes['scaler']
            filterbank = pipes['filterbank']

            # initialize dataset and dataloader
            test_dataset = H5Dataset(
                dirpath=test_dataset_dir,
                features=config.POST.FEATURES,
                labels=config.POST.LABELS,
                load=config.POST.LOAD,
                stack=config.POST.STACK,
                decimation=1,  # there must not be decimation during testing
                dct_toggle=config.POST.DCT.ON,
                n_dct=config.POST.DCT.NCOEFF,
                file_based_stats=config.POST.STANDARDIZATION.FILEBASED,
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
                cuda=config.MODEL.CUDA and not no_cuda,
            )

            # enhance mixtures for PESQ calculation
            with h5py.File(test_dataset.filepath, 'r') as f:
                n = len(f['mixtures'])
                seg_scores_i_j = np.zeros((n, 4))
                seg_scores_oracle_i_j = np.zeros((n, 4))
                for k in range(n):
                    logging.info(f'Enhancing mixture {k}/{n}...')

                    # load mixture
                    mixture = f['mixtures'][k].reshape(-1, 2)
                    foreground = f['foregrounds'][k].reshape(-1, 2)
                    background = f['backgrounds'][k].reshape(-1, 2)
                    noise = f['noises'][k].reshape(-1, 2)
                    reverb = f['reverbs'][k].reshape(-1, 2)
                    i_start, i_end = test_dataset.file_indices[k]

                    # scale signal
                    scaler.fit(mixture)
                    mixture = scaler.scale(mixture)
                    foreground = scaler.scale(foreground)
                    background = scaler.scale(background)
                    noise = scaler.scale(noise)
                    reverb = scaler.scale(reverb)
                    scaler.__init__(scaler.active)

                    # apply filterbank
                    mixture_filt = filterbank.filt(mixture)
                    foreground_filt = filterbank.filt(foreground)
                    background_filt = filterbank.filt(background)
                    noise_filt = filterbank.filt(noise)
                    reverb_filt = filterbank.filt(reverb)

                    # extract features
                    features, IRM = test_dataset[i_start:i_end]
                    features = torch.from_numpy(features).float()
                    if config.MODEL.CUDA and not no_cuda:
                        features = features.cuda()

                    # make mask prediction
                    model.eval()
                    with torch.no_grad():
                        PRM = model(features)
                        if config.MODEL.CUDA and not no_cuda:
                            PRM = PRM.cpu()
                        PRM = PRM.numpy()

                    # extrapolate predicted mask
                    PRM = wola(PRM, trim=len(mixture_filt))[:, :, np.newaxis]

                    # apply predicted mask and shadow filter
                    mixture_enhanced = filterbank.rfilt(mixture_filt*PRM)
                    foreground_enhanced = filterbank.rfilt(foreground_filt*PRM)
                    background_enhanced = filterbank.rfilt(background_filt*PRM)
                    noise_enhanced = filterbank.rfilt(noise_filt*PRM)
                    reverb_enhanced = filterbank.rfilt(reverb_filt*PRM)

                    # make reference signals
                    mixture_ref = filterbank.rfilt(mixture_filt)
                    foreground_ref = filterbank.rfilt(foreground_filt)
                    background_ref = filterbank.rfilt(background_filt)
                    noise_ref = filterbank.rfilt(noise_filt)
                    reverb_ref = filterbank.rfilt(reverb_filt)

                    # segmental SNRs
                    segSSNR, segBR, segNR, segRR = segmental_scores(
                        foreground_ref,
                        foreground_enhanced,
                        background_ref,
                        background_enhanced,
                        noise_ref,
                        noise_enhanced,
                        reverb_ref,
                        reverb_enhanced,
                    )
                    seg_scores_i_j[k, :] = segSSNR, segBR, segNR, segRR

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

                    # now repeat but with oracle mask to obtain oracle scores
                    #
                    # extrapolate oracle mask
                    IRM = wola(IRM, trim=len(mixture_filt))[:, :, np.newaxis]

                    # apply oracle mask and shadow filter
                    mixture_enhanced = filterbank.rfilt(mixture_filt*IRM)
                    foreground_enhanced = filterbank.rfilt(foreground_filt*IRM)
                    background_enhanced = filterbank.rfilt(background_filt*IRM)
                    noise_enhanced = filterbank.rfilt(noise_filt*IRM)
                    reverb_enhanced = filterbank.rfilt(reverb_filt*IRM)

                    # segmental SNRs
                    segSSNR, segBR, segNR, segRR = segmental_scores(
                        foreground_ref,
                        foreground_enhanced,
                        background_ref,
                        background_enhanced,
                        noise_ref,
                        noise_enhanced,
                        reverb_ref,
                        reverb_enhanced,
                    )
                    seg_scores_oracle_i_j[k, :] = segSSNR, segBR, segNR, segRR

                    # write oracle enhanced mixture
                    output_dir = os.path.join(model_dir, 'audio', suffix)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    sf.write(
                        os.path.join(output_dir, f'mixture_oracle_{k}.wav'),
                        mixture_enhanced*gain,
                        config.PRE.FS,
                    )

            seg_scores[i, j] = seg_scores_i_j
            seg_scores_oracle[i, j] = seg_scores_oracle_i_j
    seg_scores = np.array(seg_scores.tolist())
    seg_scores_oracle = np.array(seg_scores_oracle.tolist())

    # save MSE and segmental scores
    np.save(os.path.join(model_dir, 'mse_scores.npy'), MSE)
    np.save(os.path.join(model_dir, 'seg_scores.npy'), seg_scores)
    np.save(os.path.join(model_dir, 'seg_scores_oracle.npy'), seg_scores_oracle)

    # calculate PESQ on matlab
    try:
        import matlab
        import matlab.engine
    except Exception:
        logging.info(('Matlab engine import failed. You will have to manually '
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
    parser.add_argument('--no-cuda', action='store_true',
                        help='force testing on cpu')
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
        main(model_dir, args.force, args.no_cuda)
