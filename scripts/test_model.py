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
from brever.criterion import SNR, SISNR


def sig_figs(x, n=4):
    # significant figures
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
        x = sig_figs(x.item(), figures)
    elif isinstance(x, float):
        x = sig_figs(x, figures)
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

    # initialize model
    model = initialize_model(config)

    # load checkpoint
    checkpoint = os.path.join(args.input, 'checkpoint.pt')
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['model'])

    # disable gradients and set model to eval
    torch.set_grad_enabled(False)
    model.eval()

    # test model
    for test_path in args.test_paths:
        logging.info(f'Evaluating on {test_path}')
        test_model(model, config, test_path)


def test_model(model, config, test_path):
    # check if already tested
    scores_path = os.path.join(args.input, 'scores.json')
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            saved_scores = json.load(f)
        if test_path in saved_scores.keys() and not args.force:
            print(f'model already tested on {test_path}')
            return

    # initialize dataset
    kwargs = {}
    if hasattr(config.MODEL, 'SOURCES'):
        kwargs['components'] = config.MODEL.SOURCES
    dataset = BreverDataset(
        path=test_path,
        segment_length=0.0,
        fs=config.FS,
        model=model,
        **kwargs,
    )

    # score functions; x is the system input or output, y is the target
    score_funcs = {
        'PESQ': lambda x, y: pesq(
            config.FS, y.numpy(), x.numpy(), 'wb',
        ),
        'STOI': lambda x, y: stoi(
            y.numpy(), x.numpy(), config.FS, extended=False,
        ),
        'ESTOI': lambda x, y: stoi(
            y.numpy(), x.numpy(), config.FS, extended=True,
        ),
        'SNR': lambda x, y: -SNR()(
            x.view(1, 1, -1), y.view(1, 1, -1), [x.shape[-1]]
        ).item(),
        'SISNR': lambda x, y: -SISNR()(
            x.view(1, 1, -1), y.view(1, 1, -1), [x.shape[-1]],
        ).item(),
    }

    # init scores dict
    scores = {
        'model': {key: [] for key in score_funcs.keys()},
        'ref': {key: [] for key in score_funcs.keys()},
    }

    # main loop
    for i in range(len(dataset)):

        if i % args.verbose_period == 0:
            logging.info(f'Evaluating on mixture {i}/{len(dataset)}')

        data, target = dataset.load_segment(i)  # (2, L) and (S, 2, L)
        output = model.enhance(data, target=target)  # (L)
        target = target[0]  # (2, L)
        data = data.mean(dim=0)  # (L)
        target = target.mean(dim=0)  # (L)

        for key, score_func in score_funcs.items():
            score_model = score_func(output, target)
            score_ref = score_func(data, target)
            scores['model'][key].append(score_model)
            scores['ref'][key].append(score_ref)
            if i % args.verbose_period == 0:
                logging.info(f'{key}i: {sig_figs(score_model - score_ref)}')

        if args.output_dir is not None:
            dset_id = os.path.basename(os.path.normpath(test_path))
            input_filename = f'{dset_id}_{i:05d}_input.flac'
            output_filename = f'{dset_id}_{i:05d}_output.flac'
            input_path = os.path.join(args.output_dir, input_filename)
            output_path = os.path.join(args.output_dir, output_filename)
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
    parser.add_argument('--verbose-period', default=10,
                        help='sets frequency of log outputs')
    args = parser.parse_args()
    main()
