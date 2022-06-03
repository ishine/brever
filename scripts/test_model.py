import argparse
import logging
import os
import json

import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
import torchaudio

from brever.args import arg_type_path
from brever.config import get_config
from brever.data import BreverDataset
from brever.models import initialize_model
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


def main(test_path):
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

    # initialize model
    model = initialize_model(config)

    # initialize dataset
    dataset = BreverDataset(
        path=config.TRAINING.PATH,
        segment_length=0.0,
        fs=config.FS,
        components=config.MODEL.SOURCES,
        model=model,
    )

    # load checkpoint
    checkpoint = os.path.join(args.input, 'checkpoint.pt')
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['model'])

    # check if already tested
    scores_path = os.path.join(args.input, 'scores.json')
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            saved_scores = json.load(f)
        if test_path in saved_scores.keys() and not args.force:
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

        data, target = dataset.load_segment(i)  # (2, L) and (S, 2, L)
        output = model.enhance(data)  # (L)
        target = target[0]  # (2, L)
        data = data.mean(dim=0)  # (L)
        target = target.mean(dim=0)  # (L)

        # pesq
        pesq_model = pesq(
            config.FS,
            target.numpy(),
            output.numpy(),
            'wb',
        )
        pesq_ref = pesq(
            config.FS,
            target.numpy(),
            data.numpy(),
            'wb',
        )
        scores['model']['PESQ'].append(pesq_model)
        scores['ref']['PESQ'].append(pesq_ref)

        # stoi
        stoi_model = stoi(
            target.numpy(),
            output.numpy(),
            config.FS,
        )
        stoi_ref = stoi(
            target.numpy(),
            data.numpy(),
            config.FS,
        )
        scores['model']['STOI'].append(stoi_model)
        scores['ref']['STOI'].append(stoi_ref)

        # snr
        snr_model = -SNR()(
            output.unsqueeze(0),
            target.unsqueeze(0),
            [data.shape[-1]],
        ).item()
        snr_ref = -SNR()(
            data.unsqueeze(0),
            target.unsqueeze(0),
            [data.shape[-1]],
        ).item()
        scores['model']['SNR'].append(snr_model)
        scores['ref']['SNR'].append(snr_ref)

        logging.info(f'PESQi: {significant_figures(pesq_model - pesq_ref)}')
        logging.info(f'STOIi: {significant_figures(stoi_model - stoi_ref)}')
        logging.info(f'SNRi: {significant_figures(snr_model - snr_ref)}')

        if args.output_dir is not None:
            input_path = os.path.join(args.output_dir, f'{i:05d}_input.flac')
            output_path = os.path.join(args.output_dir, f'{i:05d}_output.flac')
            torchaudio.save(input_path, data.unsqueeze(0), config.FS)
            torchaudio.save(output_path, output.unsqueeze(0), config.FS)

    # update scores file
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            saved_scores = json.load(f)
    else:
        saved_scores = {}
    saved_scores[test_path] = format_scores(scores)
    with open(scores_path, 'w') as f:
        json.dump(saved_scores, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('input',
                        help='model directory')
    parser.add_argument('test_paths', type=arg_type_path, nargs='+',
                        help='test dataset paths')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    parser.add_argument('--output-dir',
                        help='where to write signals')
    args = parser.parse_args()

    import traceback
    for test_path in args.test_paths:
        try:
            main(test_path)
        except FileExistsError:
            traceback.print_exc()
