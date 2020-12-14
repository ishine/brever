import argparse
import os
import random
import pickle

import yaml
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

from brever.config import defaults
from brever.pytorchtools import (Feedforward, H5Dataset, TensorStandardizer,
                                 get_files_mean_and_std,
                                 StateTensorStandardizer)
from brever.display import plot_spectrogram, share_clim


def main(args):
    # seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)

    # load model configuration
    config_file = os.path.join(args.input, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    # initialize and load model
    model_args_path = os.path.join(args.input, 'model_args.yaml')
    model = Feedforward.build(model_args_path)
    state_file = os.path.join(args.input, 'checkpoint.pt')
    model.load_state_dict(torch.load(state_file, map_location='cpu'))
    if config.MODEL.CUDA:
        model = model.cuda()

    # get dataset path
    if args.dataset is None:
        # choose random snr and room
        snrs = [0, 3, 6, 9, 12, 15]
        room_aliases = [
            'surrey_room_a',
            'surrey_room_b',
            'surrey_room_c',
            'surrey_room_d',
        ]
        snr = random.choice(snrs)
        room_alias = random.choice(room_aliases)
        suffix = f'snr{snr}_room{room_alias[-1].upper()}'
        test_dataset_dir = f'{config.POST.PATH.TEST}_{suffix}'
    else:
        test_dataset_dir = args.dataset

    # load test dataset
    test_dataset = H5Dataset(
        dirpath=test_dataset_dir,
        features=config.POST.FEATURES,
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

    # set normalization
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
        stat_path = os.path.join(args.input, 'statistics.npy')
        mean, std = np.load(stat_path)
        test_dataset.transform = TensorStandardizer(mean, std)

    # load pipes
    pipes_file = os.path.join(config.POST.PATH.TRAIN, 'pipes.pkl')
    with open(pipes_file, 'rb') as f:
        pipes = pickle.load(f)
    scaler = pipes['scaler']
    filterbank = pipes['filterbank']
    framer = pipes['framer']

    # open dataset
    with h5py.File(test_dataset.filepath, 'r') as f:

        # select mixture
        if args.mix_index is None:
            # select random mixture
            n = len(f['mixtures'])
            k = random.randrange(n)
        else:
            k = args.mix_index
        print(f'Predicting mixture number {k} in {test_dataset_dir}')

        # load mixture
        mixture = f['mixtures'][k].reshape(-1, 2)
        foreground = f['foregrounds'][k].reshape(-1, 2)
        background = f['backgrounds'][k].reshape(-1, 2)

    # scale signal
    scaler.fit(mixture)
    mixture = scaler.scale(mixture)
    foreground = scaler.scale(foreground)
    background = scaler.scale(background)

    # apply filterbank
    mixture = filterbank.filt(mixture)
    foreground = filterbank.filt(foreground)
    background = filterbank.filt(background)

    # extract features and labels
    i_start, i_end = test_dataset.file_indices[k]
    features, labels = test_dataset[i_start:i_end]
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

    # grab un-normalized features
    test_dataset.transform = None
    features_raw, _ = test_dataset[i_start:i_end]

    # convert features back to numpy array
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    # frame signal
    mixture = framer.frame(mixture)
    foreground = framer.frame(foreground)
    background = framer.frame(background)

    # average channels
    mixture = mixture.mean(axis=-1)
    foreground = foreground.mean(axis=-1)
    background = background.mean(axis=-1)

    # average energy over frame samples
    mixture = (mixture**2).mean(axis=1)
    foreground = (foreground**2).mean(axis=1)
    background = (background**2).mean(axis=1)

    # convert to dB
    mixture = 10*np.log10(mixture + 1e-10)
    foreground = 10*np.log10(foreground + 1e-10)
    background = 10*np.log10(background + 1e-10)

    # plot
    vars_ = [
        'mixture',
        'foreground',
        'background',
        'features',
        'labels',
        'PRM',
    ]
    fig, axes = plt.subplots(len(vars_)//2, 2)
    axes = axes.T.flatten()
    for i, var in enumerate(vars_):
        if var == 'features':
            f = None
            set_kw = {'title': var, 'ylabel': ''}
        else:
            f = filterbank.fc
            set_kw = {'title': var}
        plot_spectrogram(
            locals()[var],
            ax=axes[i],
            fs=config.PRE.FS,
            hop_length=framer.hop_length,
            f=f,
            set_kw=set_kw,
        )

    # match signal spectrograms clims
    vars__ = ['mixture', 'foreground', 'background']
    axes_ = [axes[vars_.index(var)] for var in vars__]
    share_clim(axes_)

    # set IRM and PRM clims to 0 and 1
    vars__ = ['labels', 'PRM']
    axes_ = [axes[vars_.index(var)] for var in vars__]
    for ax in axes_:
        ax.images[0].set_clim(0, 1)

    # plot feature and label distribution
    fig, axes = plt.subplots(2, 1)
    for ax, data, title in zip(
                axes,
                [features_raw, labels],
                ['Features (not normalized)', 'Labels'],
            ):
        ax.hist(data.flatten(), bins=50)
        ax.set_title(title)
    fig.tight_layout()

    # show
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make random mask prediction')
    parser.add_argument('input', help='input model directory')
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--dataset', help='dataset directory path')
    parser.add_argument('--mix-index', type=int, help='mixture index')
    args = parser.parse_args()
    main(args)
