import argparse
import logging
import os
import json

import numpy as np
from pesq import pesq
from pystoi import stoi
import soundfile as sf
import torch

from brever.config import get_config
from brever.data import DNNDataset, ConvTasNetDataset, TensorStandardizer
from brever.models import DNN, ConvTasNet
from brever.utils import wola
from brever.logger import set_logger


def significant_figures(x, n):
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
    # check if model is trained
    loss_path = os.path.join(args.input, 'losses.npz')
    if not os.path.exists(loss_path):
        raise FileNotFoundError('model is not trained')

    # load model config
    config_path = os.path.join(args.input, 'config.yaml')
    config = get_config(config_path)

    # load datasaet config
    dset_config_path = os.path.join(args.test_path, 'config.yaml')
    dset_config = get_config(dset_config_path)

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
            framer_kwargs={
                'frame_length': config.MODEL.FRAMER.FRAME_LENGTH,
                'hop_length': config.MODEL.FRAMER.HOP_LENGTH,
            },
            filterbank_kwargs={
                'kind': config.MODEL.FILTERBANK.KIND,
                'n_filters': config.MODEL.FILTERBANK.FILTERS,
                'f_min': config.MODEL.FILTERBANK.FMIN,
                'f_max': config.MODEL.FILTERBANK.FMAX,
                'fs': config.MODEL.FILTERBANK.FS,
                'order': config.MODEL.FILTERBANK.ORDER,
            }
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
        )
        stat_path = os.path.join(args.input, 'statistics.npz')
        stats = np.load(stat_path)
        mean, std = stats['mean'], stats['std']
        model.transform = TensorStandardizer(mean, std)
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
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    # init scores dict
    scores_path = os.path.join(args.input, 'scores.json')
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)
    else:
        scores = {}

    # check if already tested
    if args.test_path in scores.keys() and not args.force:
        raise FileExistsError('model already tested on this dataset')

    scores[args.test_path] = {
        'model': {
            # 'MSE': [],
            # 'segSSNR': [],
            # 'segBR': [],
            # 'segNR': [],
            # 'segRR': [],
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
            features, labels, filt = dataset.post_proc(
                data, target, return_filter_output=True,
            )
            features = features.unsqueeze(0)

            prm = model(features).squeeze(0).numpy().T
            print(f'MSE: {np.mean((prm.T-labels.numpy())**2)}')

            prm_extra = wola(prm).T[:, np.newaxis, :data.shape[-1]]
            output = dataset.filterbank.rfilt(prm_extra*filt[:, 0, :, :])
            target = dataset.filterbank.rfilt(filt[:, 1, :, :])
        elif config.ARCH == 'convtasnet':
            data, target = dataset[i]
            output = model(data.unsqueeze(0))
            output = output.squeeze(0).numpy()
            data = data.unsqueeze(0)
            target = target[:1].numpy()
        else:
            raise ValueError(f'wrong model architecture, got {config.ARCH}')
        data = data.numpy()

        pesq_score = pesq(
            dset_config.FS,
            target.mean(axis=0),
            output.mean(axis=0),
            'wb',
        )
        scores[args.test_path]['model']['PESQ'].append(pesq_score)

        stoi_score = stoi(
            target.mean(axis=0),
            output.mean(axis=0),
            dset_config.FS,
        )
        scores[args.test_path]['model']['STOI'].append(stoi_score)

        pesq_ref = pesq(
            dset_config.FS,
            target.mean(axis=0),
            data.mean(axis=0),
            'wb',
        )
        scores[args.test_path]['ref']['PESQ'].append(pesq_ref)

        stoi_ref = stoi(
            target.mean(axis=0),
            data.mean(axis=0),
            dset_config.FS,
        )
        scores[args.test_path]['ref']['STOI'].append(stoi_ref)

        print(f'PESQi: {pesq_score - pesq_ref}')
        print(f'STOIi: {stoi_score - stoi_ref}')

        if args.output_dir is not None:
            output_path = os.path.join(args.output_dir, f'{i:05d}_output.wav')
            sf.write(output_path, output.T, dset_config.FS)

    # update scores file
    with open(scores_path, 'w') as f:
        json.dump(format_scores(scores), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('input',
                        help='model directory')
    parser.add_argument('test_path',
                        help='test dataset path')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    parser.add_argument('--output-dir',
                        help='where to write signals')
    args = parser.parse_args()
    main()
