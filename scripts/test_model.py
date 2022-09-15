import argparse
import logging
import os

import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
import torchaudio
import h5py

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
    scores_path = os.path.join(args.input, 'scores.hdf5')
    if os.path.exists(scores_path):
        with h5py.File(scores_path, 'r') as h5file:
            already_tested = test_path in h5file.keys()
        if already_tested and not args.force:
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
    score_funcs = [
        {
            'name': 'PESQ',
            'func': lambda x, y: pesq(
                config.FS, y.numpy(), x.numpy(), 'wb',
            ),
        },
        {
            'name': 'STOI',
            'func': lambda x, y: stoi(
                y.numpy(), x.numpy(), config.FS, extended=False,
            ),
        },
        {
            'name': 'ESTOI',
            'func': lambda x, y: stoi(
                y.numpy(), x.numpy(), config.FS, extended=True,
            ),
        },
        {
            'name': 'SNR',
            'func': lambda x, y: -SNR()(
                x.view(1, 1, -1), y.view(1, 1, -1), [x.shape[-1]]
            ).item(),
        },
        {
            'name': 'SISNR',
            'func': lambda x, y: -SISNR()(
                x.view(1, 1, -1), y.view(1, 1, -1), [x.shape[-1]],
            ).item(),
        },
    ]

    # initialize scores
    n_mix = len(dataset)
    dset_scores = np.empty((n_mix, len(score_funcs), 2))

    # main loop
    for i_mix in range(n_mix):

        if i_mix % args.verbose_period == 0:
            logging.info(f'Evaluating on mixture {i_mix}/{n_mix}')

        input_, target = dataset.load_segment(i_mix)  # (2, L) and (S, 2, L)
        output = model.enhance(input_, target=target)  # (L)
        target = target[0]  # (2, L)
        input_ = input_.mean(dim=0)  # (L)
        target = target.mean(dim=0)  # (L)

        for i_metric, score_func in enumerate(score_funcs):
            input_score = score_func['func'](input_, target)
            output_score = score_func['func'](output, target)
            dset_scores[i_mix, i_metric, 0] = input_score
            dset_scores[i_mix, i_metric, 1] = output_score
            if i_mix % args.verbose_period == 0:
                delta = sig_figs(output_score - input_score)
                logging.info(f"{score_func['name']}i: {delta}")

        if args.output_dir is not None:
            dset_id = os.path.basename(os.path.normpath(test_path))
            input_filename = f'{dset_id}_{i_mix:05d}_input.flac'
            output_filename = f'{dset_id}_{i_mix:05d}_output.flac'
            input_path = os.path.join(args.output_dir, input_filename)
            output_path = os.path.join(args.output_dir, output_filename)
            torchaudio.save(input_path, input_.unsqueeze(0), config.FS)
            torchaudio.save(output_path, output.unsqueeze(0), config.FS)

    # update scores file
    if os.path.exists(scores_path):
        h5file = h5py.File(scores_path, 'a')
        if test_path in h5file.keys():
            h5dset = h5file[test_path]
        else:
            h5dset = h5file.create_dataset(test_path, data=dset_scores)
    else:
        h5file = h5py.File(scores_path, 'w')
        h5file['metrics'] = [score_func['name'] for score_func in score_funcs]
        h5file['which'] = ['input', 'output']
        h5dset = h5file.create_dataset(test_path, data=dset_scores)
    h5dset.dims[0].label = 'mixture'
    h5dset.dims[1].label = 'metric'
    h5dset.dims[2].label = 'which'
    h5dset.dims[1].attach_scale(h5file['metrics'])
    h5dset.dims[2].attach_scale(h5file['which'])
    h5file.close()


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
