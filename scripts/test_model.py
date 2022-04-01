import argparse
import logging
import os
import json

import numpy as np
from pesq import pesq
from pystoi import stoi
import soundfile as sf
import torch

from brever.args import arg_type_path
from brever.config import get_config
from brever.data import DNNDataset, ConvTasNetDataset
from brever.models import DNN, ConvTasNet
from brever.logger import set_logger
from brever.training import SNR


def significant_figures(x, n=4):
    if x == 0:
        return x
    else:
        return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def format_scores(x, figures=4):
    if isinstance(x, dict):
        x = {key: format_scores(val) for key, val in x.items()}
    elif isinstance(x, list):
        x = [format_scores(val) for val in x]
    elif isinstance(x, np.floating):
        x = significant_figures(x.item(), figures)
    elif isinstance(x, float):
        x = significant_figures(x, figures)
    else:
        raise ValueError(f'got unexpected type {type(x)}')
    return x


def main():
    # check if model exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'model does not exist: {args.input}')

    # check if model is trained
    loss_path = os.path.join(args.input, 'losses.npz')
    if not os.path.exists(loss_path):
        raise FileNotFoundError('model is not trained')

    # load model config
    config_path = os.path.join(args.input, 'config.yaml')
    config = get_config(config_path)

    # initialize logger
    log_file = os.path.join(args.input, 'log.log')
    set_logger(log_file)
    logging.info(f'Testing {args.input}')
    logging.info(config.to_dict())

    # initialize dataset
    logging.info('Initializing dataset')
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
    elif config.ARCH == 'convtasnet':
        dataset = ConvTasNetDataset(
            path=args.test_path,
            components=config.MODEL.SOURCES,
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
    elif config.ARCH == 'convtasnet':
        model = ConvTasNet(
            filters=config.MODEL.ENCODER.FILTERS,
            filter_length=config.MODEL.ENCODER.FILTER_LENGTH,
            bottleneck_channels=config.MODEL.TCN.BOTTLENECK_CHANNELS,
            hidden_channels=config.MODEL.TCN.HIDDEN_CHANNELS,
            skip_channels=config.MODEL.TCN.SKIP_CHANNELS,
            kernel_size=config.MODEL.TCN.KERNEL_SIZE,
            layers=config.MODEL.TCN.LAYERS,
            repeats=config.MODEL.TCN.REPEATS,
            sources=len(config.MODEL.SOURCES),
        )
    else:
        raise ValueError(f'wrong model architecture, got {config.ARCH}')

    # load checkpoint
    checkpoint = os.path.join(args.input, 'checkpoint.pt')
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['model'])

    # check if already tested
    scores_path = os.path.join(args.input, 'scores.json')
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            saved_scores = json.load(f)
        if args.test_path in saved_scores.keys() and not args.force:
            raise FileExistsError('model already tested on this dataset')

    scores = {
        'model': {
            'PESQ': [],
            'STOI': [],
            'SNR': [],
        },
        'ref': {
            'PESQ': [],
            'STOI': [],
            'SNR': [],
        }
    }

    # disable gradients and set model to eval
    torch.set_grad_enabled(False)
    model.eval()

    # main loop
    for i in range(len(dataset)):

        logging.info(f'Evaluating on mixture {i}/{len(dataset)}')

        if config.ARCH == 'dnn':
            data, target = dataset.load_segment(i)
            output, mask = model.enhance(data, dataset, True)
            target = target[0]
        elif config.ARCH == 'convtasnet':
            data, target = dataset[i]
            output = model(data.unsqueeze(0))
            output = output.squeeze(0)
            data = data.unsqueeze(0)
            target = target[:1]
        else:
            raise ValueError(f'wrong model architecture, got {config.ARCH}')
        output = output.numpy()
        target = target.numpy()
        data = data.numpy()

        # pesq
        pesq_model = pesq(
            config.FS,
            target.mean(axis=0),
            output.mean(axis=0),
            'wb',
        )
        pesq_ref = pesq(
            config.FS,
            target.mean(axis=0),
            data.mean(axis=0),
            'wb',
        )
        scores['model']['PESQ'].append(pesq_model)
        scores['ref']['PESQ'].append(pesq_ref)

        # stoi
        stoi_model = stoi(
            target.mean(axis=0),
            output.mean(axis=0),
            config.FS,
        )
        stoi_ref = stoi(
            target.mean(axis=0),
            data.mean(axis=0),
            config.FS,
        )
        scores['model']['STOI'].append(stoi_model)
        scores['ref']['STOI'].append(stoi_ref)

        # snr
        snr_model = -SNR()(
            torch.from_numpy(output.copy()),
            torch.from_numpy(target.copy()),
        ).item()
        snr_ref = -SNR()(
            torch.from_numpy(data.copy()),
            torch.from_numpy(target.copy()),
        ).item()
        scores['model']['SNR'].append(snr_model)
        scores['ref']['SNR'].append(snr_ref)

        logging.info(f'PESQi: {significant_figures(pesq_model - pesq_ref)}')
        logging.info(f'STOIi: {significant_figures(stoi_model - stoi_ref)}')
        logging.info(f'SNRi: {significant_figures(snr_model - snr_ref)}')

        if args.output_dir is not None:
            output_path = os.path.join(args.output_dir, f'{i:05d}_output.flac')
            sf.write(output_path, output.T, config.FS)

    # update scores file
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            saved_scores = json.load(f)
    else:
        saved_scores = {}
    saved_scores[args.test_path] = format_scores(scores)
    with open(scores_path, 'w') as f:
        json.dump(saved_scores, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('input',
                        help='model directory')
    parser.add_argument('test_path', type=arg_type_path,
                        help='test dataset path')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    parser.add_argument('--output-dir',
                        help='where to write signals')
    args = parser.parse_args()
    main()
