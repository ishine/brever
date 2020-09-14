import os
import argparse
from glob import glob
import pickle
import logging
import sys

import yaml
import torch
import numpy as np
import h5py
import matlab
import matlab.engine
import soundfile as sf

from brever.config import defaults
from brever.utils import wola
from brever.pytorchtools import (Feedforward, H5Dataset, TensorStandardizer,
                                 get_mean_and_std, evaluate)
from brever.modelmanagement import get_feature_indices, get_file_indices


def main(model_dir, force):
    logging.info(f'Processing {model_dir}')

    # check if model is already tested
    output_pesq_path = os.path.join(model_dir, 'eval_PESQ.npy')
    output_mse_path = os.path.join(model_dir, 'eval_MSE.npy')
    if os.path.exists(output_pesq_path) and os.path.exists(output_mse_path):
        if not force:
            logging.info(f'Model is already tested!')
            return

    # check if model is trained
    train_losses_path = os.path.join(model_dir, 'train_losses.npy')
    val_losses_path = os.path.join(model_dir, 'val_losses.npy')
    if os.path.exists(train_losses_path) and os.path.exists(val_losses_path):
        pass
    else:
        logging.info(f'Model is not trained!')
        return

    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    # seed for reproducibility
    torch.manual_seed(0)

    # initialize training dataset; it is not used but we need its n_features
    # and n_labels attributes as well as its mean and std
    train_dataset_path = os.path.join(config.POST.PATH.TRAIN, 'dataset.hdf5')
    feature_indices = get_feature_indices(config.POST.PATH.TRAIN,
                                          config.POST.FEATURES)
    file_indices = get_file_indices(config.POST.PATH.TRAIN)
    train_dataset = H5Dataset(
        filepath=train_dataset_path,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        decimation=config.POST.DECIMATION,
        feature_indices=feature_indices,
        file_indices=file_indices,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.MODEL.BATCHSIZE,
        shuffle=config.MODEL.SHUFFLE,
        num_workers=config.MODEL.NWORKERS,
        drop_last=True,
    )

    # get mean and std
    if config.POST.GLOBALSTANDARDIZATION:
        stat_path = os.path.join(model_dir, 'statistics.npy')
        if os.path.exists(stat_path):
            logging.info('Loading mean and std...')
            mean, std = np.load(stat_path)
        else:
            logging.info('Calculating mean and std...')
            mean, std = get_mean_and_std(train_dataloader, config.POST.LOAD)

    # initialize and load network
    logging.info('Loading model...')
    model = Feedforward(
        input_size=train_dataset.n_features,
        output_size=train_dataset.n_labels,
        n_layers=config.MODEL.NLAYERS,
        dropout_toggle=config.MODEL.DROPOUT.ON,
        dropout_rate=config.MODEL.DROPOUT.RATE,
        batchnorm_toggle=config.MODEL.BATCHNORM.ON,
        batchnorm_momentum=config.MODEL.BATCHNORM.MOMENTUM,
    )
    if config.MODEL.CUDA:
        model = model.cuda()
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

    # load matlab engine
    logging.info('Starting MATLAB engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath('matlab\\loizou', nargout=0)

    # main loop
    logging.info('Starting main loop:')
    if 'onlyreverb' in config.POST.PATH.TEST:
        snrs = [0]
    else:
        snrs = [0, 3, 6, 9, 12, 15]
    if 'onlydiffuse' in config.POST.PATH.TEST:
        room_aliases = ['surrey_anechoic']
    else:
        room_aliases = [
            'surrey_room_a',
            'surrey_room_b',
            'surrey_room_c',
            'surrey_room_d',
        ]
    PESQ = np.zeros((len(snrs), len(room_aliases)))
    MSE = np.zeros((len(snrs), len(room_aliases)))
    for i, snr in enumerate(snrs):
        for j, room_alias in enumerate(room_aliases):
            if 'onlydiffuse' in config.POST.PATH.TEST:
                suffix = f'snr{snr}_anechoic'
            else:
                suffix = f'snr{snr}_room{room_alias[-1].upper()}'
            test_dataset_dir = f'{config.POST.PATH.TEST}_{suffix}'
            test_dataset_path = os.path.join(test_dataset_dir, 'dataset.hdf5')
            logging.info(f'Processing {test_dataset_dir}:')

            # recalculate feature_indices and file_indices
            feature_indices = get_feature_indices(test_dataset_dir,
                                                  config.POST.FEATURES)
            file_indices = get_file_indices(test_dataset_dir)
            test_dataset = H5Dataset(
                filepath=test_dataset_path,
                load=config.POST.LOAD,
                stack=config.POST.STACK,
                # decimation=config.POST.DECIMATION,
                feature_indices=feature_indices,
                file_indices=file_indices,
            )
            test_dataloader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=config.MODEL.BATCHSIZE,
                shuffle=config.MODEL.SHUFFLE,
                num_workers=config.MODEL.NWORKERS,
                drop_last=True,
            )
            if config.POST.GLOBALSTANDARDIZATION:
                test_dataset.transform = TensorStandardizer(mean, std)
            logging.info(f'Calculating MSE...')
            MSE[i, j] = evaluate(
                model=model,
                criterion=criterion,
                dataloader=test_dataloader,
                load=config.POST.LOAD,
                cuda=config.MODEL.CUDA,
            )

            # calculate PESQ; first load mixtures
            dpesqs = []
            with h5py.File(test_dataset_path, 'r') as f:
                n = len(f['mixtures'])
                for k in range(n):
                    logging.info(f'Calculating PESQ for mixture {k}/{n}...')
                    mixture = f['mixtures'][k].reshape(-1, 2)
                    foreground = f['foregrounds'][k].reshape(-1, 2)
                    i_start, i_end = file_indices[k]

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

                    # make RM prediction
                    model.eval()
                    with torch.no_grad():
                        PRM = model(features)
                        if config.MODEL.CUDA:
                            PRM = PRM.cpu()
                        PRM = PRM.numpy()

                    # apply predicted RM
                    PRM_ = wola(PRM, trim=len(mixture_filt))[:, :, np.newaxis]
                    mixture_enhanced = filterbank.rfilt(mixture_filt*PRM_)
                    mixture_ref = filterbank.rfilt(mixture_filt)
                    foreground_ref = filterbank.rfilt(foreground_filt)

                    # write audio
                    gain = 1/mixture.max()
                    output_dir = os.path.join(model_dir, 'audio', suffix)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    sf.write(os.path.join(output_dir,
                                          f'mixture_enhanced_{k}.wav'),
                             mixture_enhanced*gain, config.PRE.FS)

                    # remove noise-only parts
                    npad = round(config.PRE.MIXTURES.PADDING*config.PRE.FS)
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
                                           config.PRE.FS)
                    pesq_after = eng.pesq(foreground_ref, mixture_enhanced,
                                          config.PRE.FS)
                    dpesq = pesq_after - pesq_before
                    dpesqs.append(dpesq)
                    logging.info(f'Delta PESQ: {dpesq:.2f}')

            PESQ[i, j] = np.mean(dpesqs)

    np.save(os.path.join(model_dir, 'eval_MSE.npy'), MSE)
    np.save(os.path.join(model_dir, 'eval_PESQ.npy'), PESQ)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('input',
                        help='input model directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    for model_dir in glob(args.input):
        main(model_dir, args.force)
