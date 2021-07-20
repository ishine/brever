import argparse
import os
import random
import pickle
import re

import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io

from brever.config import defaults
import brever.pytorchtools as bptt
from brever.display import plot_spectrogram, share_clim
import brever.modelmanagement as bmm


def main(args):
    # seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)

    # calculate subplot grid
    n_rows = len(args.input)
    n_cols = 2
    n_models = len(args.input)
    fig_run, axes_run = plt.subplots(n_rows, n_cols)

    for i_model, model_dirpath in enumerate(args.input):
        # load model configuration
        config = defaults()
        config_file = os.path.join(model_dirpath, 'config.yaml')
        config.update(bmm.read_yaml(config_file))

        # initialize and load model
        model_args_path = os.path.join(model_dirpath, 'model_args.yaml')
        model = bptt.Feedforward.build(model_args_path)
        state_file = os.path.join(model_dirpath, 'checkpoint.pt')
        model.load_state_dict(torch.load(state_file, map_location='cpu'))
        if config.MODEL.CUDA and not args.no_cuda:
            model = model.cuda()

        # get dataset path
        snrs = [0, 3, 6, 9, 12, 15]
        room_aliases = [
            'surrey_room_a',
            'surrey_room_b',
            'surrey_room_c',
            'surrey_room_d',
        ]
        if args.dataset is None:
            # choose random snr and room
            snr = random.choice(snrs)
            room_alias = random.choice(room_aliases)
            suffix = f'snr{snr}_room{room_alias[-1].upper()}'
            test_dataset_dir = f'{config.POST.PATH.TEST}_{suffix}'
        else:
            test_dataset_dir = args.dataset
            m = re.match('.+snr(.{1,2})_room(.)', test_dataset_dir)
            snr = int(m.group(1))
            room_alias = f'surrey_room_{m.group(2).lower()}'

        # load test dataset
        test_dataset = bptt.H5Dataset(
            dirpath=test_dataset_dir,
            features=config.POST.FEATURES,
            labels=config.POST.LABELS,
            load=config.POST.LOAD,
            stack=config.POST.STACK,
            decimation=1,  # there must not be decimation during testing
            dct_toggle=config.POST.DCT.ON,
            n_dct=config.POST.DCT.NCOEFF,
            prestack=config.POST.PRESTACK,
        )

        # set normalization
        if config.POST.NORMALIZATION.TYPE == 'global':
            stat_path = os.path.join(model_dirpath, 'statistics.npy')
            mean, std = np.load(stat_path)
            test_dataset.transform = bptt.TensorStandardizer(mean, std)
        elif config.POST.NORMALIZATION.TYPE == 'recursive':
            stat_path = os.path.join(model_dirpath, 'statistics.npy')
            mean, std = np.load(stat_path)
            test_dataset.transform = bptt.ResursiveTensorStandardizer(
                mean=mean,
                std=std,
                momentum=config.POST.NORMALIZATION.RECURSIVEMOMENTUM,
            )
        elif config.POST.NORMALIZATION.TYPE == 'filebased':
            test_means, test_stds = bptt.get_files_mean_and_std(
                test_dataset,
                config.POST.NORMALIZATION.UNIFORMFEATURES,
            )
            test_dataset.transform = bptt.StateTensorStandardizer(
                test_means,
                test_stds,
            )
        else:
            raise ValueError('Unrecognized normalization strategy: '
                             f'{config.POST.NORMALIZATION.TYPE}')

        # load pipes
        pipes_file = os.path.join(config.POST.PATH.TRAIN, 'pipes.pkl')
        with open(pipes_file, 'rb') as f:
            pipes = pickle.load(f)
        scaler = pipes['scaler']
        filterbank = pipes['filterbank']
        framer = pipes['framer']

        # load pesq score
        pesq_filepath = os.path.join(model_dirpath, 'pesq_scores.mat')
        pesq = scipy.io.loadmat(pesq_filepath)['scores']

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

            # grab pesq
            pesq = pesq[snrs.index(snr), room_aliases.index(room_alias), k]

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
        if config.MODEL.CUDA and not args.no_cuda:
            features = features.cuda()

        # make RM prediction
        model.eval()
        with torch.no_grad():
            PRM = model(features)
            if config.MODEL.CUDA and not args.no_cuda:
                PRM = PRM.cpu()
            PRM = PRM.numpy()
        mse = np.mean((PRM-labels)**2)

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
        mixture = 10*np.log10(mixture + np.nextafter(0, 1))
        foreground = 10*np.log10(foreground + np.nextafter(0, 1))
        background = 10*np.log10(background + np.nextafter(0, 1))

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
            elif var == 'PRM':
                f = filterbank.fc
                title = f'PRM, dPESQ: {pesq:.2f}, MSE: {mse*1e3:.2f}e-3'
                set_kw = {'title': title}
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

        # plot on running fig
        plot_spectrogram(
            labels,
            ax=axes_run[i_model, 0],
            fs=config.PRE.FS,
            hop_length=framer.hop_length,
            f=filterbank.fc,
            set_kw={'title': 'labels'},
        )
        model_id = os.path.basename(os.path.dirname(model_dirpath))
        title = f'{model_id[:6]}..., dPESQ: {pesq:.2f}, MSE: {mse*1e3:.2f}e-3'
        set_kw = {'title': title}
        plot_spectrogram(
            PRM,
            ax=axes_run[i_model, 1],
            fs=config.PRE.FS,
            hop_length=framer.hop_length,
            f=filterbank.fc,
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

    # # plot feature and label distribution
    # fig, axes = plt.subplots(2, 1)
    # for ax, data, title in zip(
    #             axes,
    #             [features_raw, labels],
    #             ['Features (not normalized)', 'Labels'],
    #         ):
    #     ax.hist(data.flatten(), bins=50)
    #     ax.set_title(title)
    # fig.tight_layout()

    # show
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make random mask prediction')
    parser.add_argument('input', nargs='+', help='input model directories')
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--dataset', help='dataset directory path')
    parser.add_argument('--mix-index', type=int, help='mixture index')
    parser.add_argument('--no-cuda', action='store_true',
                        help='force model loading on cpu')
    args = parser.parse_args()
    main(args)
