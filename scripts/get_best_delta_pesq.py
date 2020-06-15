import os
import pickle

import numpy as np
import matlab.engine
import yaml
import h5py
import torch

from brever.config import defaults
from brever.utils import wola
from brever.modelmanagement import get_file_indices, get_feature_indices
from brever.pytorchtools import H5Dataset, Feedforward, TensorStandardizer


if __name__ == '__main__':
    models_dir = 'models'
    best_pesq = 0
    best_model = None
    best_snr_i, best_room_j = None, None
    for model_id in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_id)
        pesq_filepath = os.path.join(model_dir, 'eval_PESQ.npy')
        if os.path.exists(pesq_filepath):
            pesqs = np.load(pesq_filepath)
            max_pesq_i = pesqs.argmax()
            max_pesq_i = np.unravel_index(max_pesq_i, pesqs.shape)
            max_pesq = pesqs[max_pesq_i]
            if max_pesq > best_pesq:
                best_pesq = max_pesq
                best_model = model_id
                best_snr_i, best_room_j = max_pesq_i

    snrs = [0, 3, 6, 9, 12, 15]
    room_aliases = [
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
    ]
    snr = snrs[best_snr_i]
    room_alias = room_aliases[best_room_j]
    best_condition = f'snr{snr}_room{room_alias[-1].upper()}'

    print(f'Best average score: {best_pesq:.2f}')
    print(f'Achieved by: {best_model}')
    print(f'In condition: {best_condition}')

    # start matlab engine
    print('Starting MATLAB engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath('matlab\\loizou', nargout=0)

    # load config file
    best_model_dir = os.path.join(models_dir, best_model)
    config_file = os.path.join(best_model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    # load pipes
    print('Loading pipes...')
    pipes_file = os.path.join(config.POST.PATH.TRAIN, 'pipes.pkl')
    with open(pipes_file, 'rb') as f:
        pipes = pickle.load(f)
    scaler = pipes['scaler']
    filterbank = pipes['filterbank']

    # get mean and std
    stat_path = os.path.join(best_model_dir, 'statistics.npy')
    print('Loading mean and std...')
    mean, std = np.load(stat_path)

    # initialize dataset
    test_dataset_dir = f'{config.POST.PATH.TEST}_{best_condition}'
    test_dataset_path = os.path.join(test_dataset_dir, 'dataset.hdf5')
    file_indices = get_file_indices(test_dataset_dir)
    feature_indices = get_feature_indices(test_dataset_dir,
                                          sorted(config.POST.FEATURES))
    test_dataset = H5Dataset(
        filepath=test_dataset_path,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        # decimation=config.POST.DECIMATION,
        feature_indices=feature_indices,
        file_indices=file_indices,
    )
    if config.POST.GLOBALSTANDARDIZATION:
        test_dataset.transform = TensorStandardizer(mean, std)

    # initialize and load network
    print('Loading model...')
    model = Feedforward(
        input_size=test_dataset.n_features,
        output_size=test_dataset.n_labels,
        n_layers=config.MODEL.NLAYERS,
        dropout_toggle=config.MODEL.DROPOUT.ON,
        dropout_rate=config.MODEL.DROPOUT.RATE,
        batchnorm_toggle=config.MODEL.BATCHNORM.ON,
        batchnorm_momentum=config.MODEL.BATCHNORM.MOMENTUM,
    )
    if config.MODEL.CUDA:
        model = model.cuda()
    state_file = os.path.join(best_model_dir, 'checkpoint.pt')
    model.load_state_dict(torch.load(state_file))

    # calculate PESQ; first load mixtures
    dpesqs = []
    with h5py.File(test_dataset_path, 'r') as f:
        n = len(f['mixtures'])
        for k in range(n):
            print(f'Calculating PESQ for mixture {k}/{n}...')
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
            print(f'Delta PESQ: {dpesq:.2f}')
