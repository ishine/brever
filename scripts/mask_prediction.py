import argparse
import logging
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from brever.config import get_config
from brever.data import DNNDataset
from brever.models import DNN
import brever.display as bplot


def main(args):
    # seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)

    # load model config
    config_path = os.path.join(args.input, 'config.yaml')
    config = get_config(config_path)

    # initialize dataset
    if config.ARCH == 'dnn':
        dataset = DNNDataset(
            path=args.test_path,
            features=config.MODEL.FEATURES,
            stacks=config.MODEL.STACKS,
            decimation=1,
            stft_frame_length=config.MODEL.STFT.FRAME_LENGTH,
            stft_hop_length=config.MODEL.STFT.HOP_LENGTH,
            stft_window=config.MODEL.STFT.WINDOW,
            mel_filters=config.MODEL.MEL_FILTERS,
            fs=config.FS,
        )
    else:
        raise ValueError(f'wrong model architecture, got {config.ARCH}')

    # initialize model
    logging.info('Initializing model')
    if config.ARCH == 'dnn':
        model = DNN(
            input_size=dataset.n_features,
            output_size=dataset.n_labels,
            hidden_layers=config.MODEL.HIDDEN_LAYERS,
            dropout=config.MODEL.DROPOUT,
            batchnorm=config.MODEL.BATCH_NORM.TOGGLE,
            batchnorm_momentum=config.MODEL.BATCH_NORM.MOMENTUM,
            normalization=config.MODEL.NORMALIZATION.TYPE,
        )
    else:
        raise ValueError(f'wrong model architecture, got {config.ARCH}')

    # load checkpoint
    checkpoint = os.path.join(args.input, 'checkpoint.pt')
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['model'])

    # disable gradients and set model to eval
    torch.set_grad_enabled(False)
    model.eval()

    j = args.mix_index if args.mix_index is not None else 0
    if config.ARCH == 'dnn':
        mixture, target = dataset.load_segment(j)
        features, labels = dataset[j]
        output, prediction = model.enhance(mixture, dataset, True)
        foreground, background = target
        mixture, _ = dataset.stft.analyze(mixture.unsqueeze(0))
        foreground, _ = dataset.stft.analyze(foreground.unsqueeze(0))
        background, _ = dataset.stft.analyze(background.unsqueeze(0))
    else:
        raise ValueError(f'wrong model architecture, got {config.ARCH}')
    mixture = mixture[0, 0].log10()
    foreground = foreground[0, 0].log10()
    background = background[0, 0].log10()
    features = features[:64, :]

    # plot
    vars_ = [
        'mixture',
        'foreground',
        'background',
        'features',
        'labels',
        'prediction',
    ]
    fig, axes = plt.subplots(len(vars_)//2, 2)
    axes = axes.T.flatten()
    for i, var in enumerate(vars_):
        if var == 'prediction' or var == 'labels' or var == 'features':
            f = dataset.mel_fb.fc.numpy()
            print(f)
            set_kw = {'title': var}
        elif var:
            n_fft = dataset.stft.frame_length
            f = np.linspace(0, config.FS/2, n_fft//2+1)
            set_kw = {'title': var}
        bplot.plot_spectrogram(
            locals()[var].T,
            ax=axes[i],
            fs=config.FS,
            hop_length=dataset.stft.hop_length,
            f=f,
            set_kw=set_kw,
        )

    # match signal spectrograms clims
    vars__ = ['mixture', 'foreground', 'background']
    axes_ = [axes[vars_.index(var)] for var in vars__]
    bplot.share_clim(axes_)

    # set IRM and PRM clims to 0 and 1
    vars__ = ['labels', 'prediction']
    axes_ = [axes[vars_.index(var)] for var in vars__]
    for ax in axes_:
        ax.images[0].set_clim(0, 1)

    # show
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make random mask prediction')
    parser.add_argument('input', help='model directory')
    parser.add_argument('test_path', help='dataset directory')
    parser.add_argument('--seed', type=int, help='seed', default=0)
    parser.add_argument('--mix-index', type=int, help='mixture index')
    args = parser.parse_args()
    main(args)
