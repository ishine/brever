import argparse
import os
import pickle
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from brever.config import defaults
import brever.data as bdata
import brever.display as bplot
import brever.management as bm
from brever.models import DNN


def main(args):
    # seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)

    # calculate subplot grid
    n_rows = len(args.input)
    n_cols = 2
    n_models = len(args.input)
    fig_run, axes_run = plt.subplots(n_rows, n_cols)
    if axes_run.ndim == 1:
        axes_run = axes_run.reshape(1, len(axes_run))

    for i_model, model_dir in enumerate(args.input):
        # load model configuration
        config = defaults()
        config_file = os.path.join(model_dir, 'config.yaml')
        config.update(bm.read_yaml(config_file))

        # initialize and load model
        model_args_path = os.path.join(model_dir, 'model_args.yaml')
        model_args = bm.read_yaml(model_args_path)
        model = DNN(**model_args)
        state_file = os.path.join(model_dir, 'checkpoint.pt')
        model.load_state_dict(torch.load(state_file, map_location='cpu'))

        # get dataset path
        if args.dataset is None:
            # choose random data in the list of test paths
            test_dir = random.choice(sorted(config.POST.PATH.TEST))
        else:
            test_dir = args.dataset

        # load test dataset
        test_dataset = bdata.H5Dataset(
            dirpath=test_dir,
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
            stat_path = os.path.join(model_dir, 'statistics.npy')
            mean, std = np.load(stat_path)
            test_dataset.transform = bdata.TensorStandardizer(mean, std)
        elif config.POST.NORMALIZATION.TYPE == 'recursive':
            stat_path = os.path.join(model_dir, 'statistics.npy')
            mean, std = np.load(stat_path)
            test_dataset.transform = bdata.ResursiveTensorStandardizer(
                mean=mean,
                std=std,
                momentum=config.POST.NORMALIZATION.RECURSIVEMOMENTUM,
            )
        elif config.POST.NORMALIZATION.TYPE == 'filebased':
            test_means, test_stds = bdata.get_files_mean_and_std(
                test_dataset,
                config.POST.NORMALIZATION.UNIFORMFEATURES,
            )
            test_dataset.transform = bdata.StateTensorStandardizer(
                test_means,
                test_stds,
            )
        else:
            raise ValueError('Unrecognized normalization strategy: '
                             f'{config.POST.NORMALIZATION.TYPE}')

        # load pipes
        pipes_file = os.path.join(test_dir, 'pipes.pkl')
        with open(pipes_file, 'rb') as f:
            pipes = pickle.load(f)
        scaler = pipes['scaler']
        filterbank = pipes['filterbank']
        framer = pipes['framer']

        # load scores
        scores_file = os.path.join(model_dir, 'scores.json')
        scores = bm.read_json(scores_file)

        # open dataset
        with h5py.File(test_dataset.filepath, 'r') as f:

            # select mixture
            if args.mix_index is None:
                # select random mixture
                n = len(f['mixture'])
                k = random.randrange(n)
            else:
                k = args.mix_index
            print(f'Predicting mixture number {k} in {test_dir}')

            # grab pesq
            pesq_in = scores[test_dir]['ref']['PESQ'][k]
            pesq_out = scores[test_dir]['model']['PESQ'][k]
            dpesq = pesq_out - pesq_in

            # load mixture
            mixture = f['mixture'][k].reshape(-1, 2)
            foreground = f['foreground'][k].reshape(-1, 2)
            # background = f['background'][k].reshape(-1, 2)
            background = foreground

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

        # make RM prediction
        model.eval()
        with torch.no_grad():
            PRM = model(features)
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
                title = f'dPESQ: {dpesq:.2f}, MSE: {mse*1e3:.2f}e-3'
                set_kw = {'title': title}
            else:
                f = filterbank.fc
                set_kw = {'title': var}
            bplot.plot_spectrogram(
                locals()[var],
                ax=axes[i],
                fs=config.PRE.FS,
                hop_length=framer.hop_length,
                f=f,
                set_kw=set_kw,
            )

        # plot on running fig
        bplot.plot_spectrogram(
            labels,
            ax=axes_run[i_model, 0],
            fs=config.PRE.FS,
            hop_length=framer.hop_length,
            f=filterbank.fc,
            set_kw={'title': 'labels'},
        )
        title = f'{model_dir}, dPESQ: {dpesq:.2f}, MSE: {mse*1e3:.2f}e-3'
        set_kw = {'title': title}
        bplot.plot_spectrogram(
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
        bplot.share_clim(axes_)

        # set IRM and PRM clims to 0 and 1
        vars__ = ['labels', 'PRM']
        axes_ = [axes[vars_.index(var)] for var in vars__]
        for ax in axes_:
            ax.images[0].set_clim(0, 1)

    # show
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make random mask prediction')
    parser.add_argument('input', nargs='+', help='input model directories')
    parser.add_argument('--seed', type=int, help='seed', default=0)
    parser.add_argument('--dataset', help='dataset directory path')
    parser.add_argument('--mix-index', type=int, help='mixture index')
    args = parser.parse_args()
    main(args)
