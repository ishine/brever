import os
import argparse
from glob import glob
import pickle
import logging
import sys
import time

import yaml
import numba  # noqa: F401
import torch
import numpy as np
import h5py
import soundfile as sf
from pesq import pesq
from pystoi import stoi

from brever.config import defaults
from brever.utils import wola, segmental_scores
import brever.pytorchtools as bptt
from brever.modelmanagement import globbed


def main(model_dir, args):
    logging.info(f'Processing {model_dir}')

    # check if model is already tested
    output_mat_path = os.path.join(model_dir, 'scores.mat')
    output_npz_path = os.path.join(model_dir, 'scores.npz')
    if os.path.exists(output_mat_path) and os.path.exists(output_npz_path):
        if not args.force:
            logging.info('Model is already tested!')
            return

    # check if model is trained
    loss_path = os.path.join(model_dir, 'losses.npz')
    if not os.path.exists(loss_path):
        logging.info('Model is not trained!')
        return

    # load config file
    config = defaults()
    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        config.update(yaml.safe_load(f))

    # seed for reproducibility
    torch.manual_seed(0)

    # get mean and std
    if config.POST.STANDARDIZATION.FILEBASED:
        pass
    else:
        stat_path = os.path.join(model_dir, 'statistics.npy')
        logging.info('Loading mean and std...')
        mean, std = np.load(stat_path)

    # initialize and load network
    logging.info('Loading model...')
    model_args_path = os.path.join(model_dir, 'model_args.yaml')
    model = bptt.Feedforward.build(model_args_path)
    state_file = os.path.join(model_dir, 'checkpoint.pt')
    model.load_state_dict(torch.load(state_file, map_location='cpu'))
    if config.MODEL.CUDA and not args.no_cuda:
        model = model.cuda()

    # initialize criterion
    criterion = getattr(torch.nn, config.MODEL.CRITERION)()

    # main loop
    enhancement_time = 0
    n_mixtures_enhanced = 0
    scores = {}
    for test_dir in globbed(config.POST.PATH.TEST):
        # verbose and initialize scores field
        logging.info(f'Processing {test_dir}:')
        scores[test_dir] = {
            'enhanced': {
                'segSSNR': [],
                'segBR': [],
                'segNR': [],
                'segRR': [],
                'PESQ': [],
                'STOI': [],
            },
            'oracle': {
                'segSSNR': [],
                'segBR': [],
                'segNR': [],
                'segRR': [],
                'PESQ': [],
                'STOI': [],
            },
            'ref': {
                'segSSNR': [],
                'segBR': [],
                'segNR': [],
                'segRR': [],
                'PESQ': [],
                'STOI': [],
            }
        }

        # load pipes
        logging.info('Loading pipes...')
        pipes_file = os.path.join(test_dir, 'pipes.pkl')
        with open(pipes_file, 'rb') as f:
            pipes = pickle.load(f)
        scaler = pipes['scaler']
        filterbank = pipes['filterbank']

        # initialize dataset and dataloader
        test_dataset = bptt.H5Dataset(
            dirpath=test_dir,
            features=config.POST.FEATURES,
            labels=config.POST.LABELS,
            load=config.POST.LOAD,
            stack=config.POST.STACK,
            decimation=1,  # there must not be decimation during testing
            dct_toggle=config.POST.DCT.ON,
            n_dct=config.POST.DCT.NCOEFF,
            file_based_stats=config.POST.STANDARDIZATION.FILEBASED,
            prestack=config.POST.PRESTACK,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=config.MODEL.BATCHSIZE,
            shuffle=config.MODEL.SHUFFLE,
            num_workers=config.MODEL.NWORKERS,
            drop_last=True,
        )

        # set normalization transform
        if config.POST.STANDARDIZATION.FILEBASED:
            test_means, test_stds = bptt.get_files_mean_and_std(
                test_dataset,
                config.POST.STANDARDIZATION.UNIFORMFEATURES,
            )
            test_dataset.transform = bptt.StateTensorStandardizer(
                test_means,
                test_stds,
            )
        else:
            test_dataset.transform = bptt.TensorStandardizer(mean, std)

        # calculate MSE
        logging.info('Calculating MSE...')
        scores[test_dir]['enhanced']['mse'] = bptt.evaluate(
            model=model,
            criterion=criterion,
            dataloader=test_dataloader,
            cuda=config.MODEL.CUDA and not args.no_cuda,
        )

        # open hdf5 file to load mixtures for objective metrics calculation
        h5f = h5py.File(test_dataset.filepath, 'r')

        # loop over mixtures
        n = len(h5f['mixture'])
        for k in range(n):
            start_time = time.time()
            if k == 0:
                logging.info(f'Enhancing mixture {k}/{n}...')
            else:
                time_per_mix = enhancement_time/n_mixtures_enhanced
                logging.info(f'Enhancing mixture {k}/{n}... '
                             f'Average enhancement time: '
                             f'{time_per_mix:.2f}')

            # load mixture
            mixture = h5f['mixture'][k].reshape(-1, 2)
            foreground = h5f['foreground'][k].reshape(-1, 2)
            background = h5f['background'][k].reshape(-1, 2)
            noise = h5f['noise'][k].reshape(-1, 2)
            reverb = h5f['late_target'][k].reshape(-1, 2)
            i_start, i_end = test_dataset.file_indices[k]

            # scale signal
            scaler.fit(mixture)
            mixture = scaler.scale(mixture)
            foreground = scaler.scale(foreground)
            background = scaler.scale(background)
            noise = scaler.scale(noise)
            reverb = scaler.scale(reverb)
            scaler.__init__(scaler.active)

            # apply filterbank
            mixture_filt = filterbank.filt(mixture)
            foreground_filt = filterbank.filt(foreground)
            background_filt = filterbank.filt(background)
            noise_filt = filterbank.filt(noise)
            reverb_filt = filterbank.filt(reverb)

            # extract features
            features, IRM = test_dataset[i_start:i_end]
            features = torch.from_numpy(features).float()
            if config.MODEL.CUDA and not args.no_cuda:
                features = features.cuda()

            # make mask prediction
            model.eval()
            with torch.no_grad():
                PRM = model(features)
                if config.MODEL.CUDA and not args.no_cuda:
                    PRM = PRM.cpu()
                PRM = PRM.numpy()

            # extrapolate predicted mask
            PRM = wola(PRM, trim=len(mixture_filt))[:, :, np.newaxis]

            # apply predicted mask and reverse filter
            mixture_enhanced = filterbank.rfilt(mixture_filt*PRM)
            foreground_enhanced = filterbank.rfilt(foreground_filt*PRM)
            background_enhanced = filterbank.rfilt(background_filt*PRM)
            noise_enhanced = filterbank.rfilt(noise_filt*PRM)
            reverb_enhanced = filterbank.rfilt(reverb_filt*PRM)

            # load reference signals
            mixture_ref = h5f['mixture_ref'][k].reshape(-1, 2)
            foreground_ref = h5f['foreground_ref'][k].reshape(-1, 2)
            background_ref = h5f['background_ref'][k].reshape(-1, 2)
            noise_ref = h5f['noise_ref'][k].reshape(-1, 2)
            reverb_ref = h5f['late_target_ref'][k].reshape(-1, 2)

            # segmental SNRs
            segSSNR, segBR, segNR, segRR = segmental_scores(
                foreground_ref,
                foreground_enhanced,
                background_ref,
                background_enhanced,
                noise_ref,
                noise_enhanced,
                reverb_ref,
                reverb_enhanced,
            )
            scores[test_dir]['enhanced']['segSSNR'].append(segSSNR)
            scores[test_dir]['enhanced']['segBR'].append(segBR)
            scores[test_dir]['enhanced']['segNR'].append(segNR)
            scores[test_dir]['enhanced']['segRR'].append(segRR)

            # write mixtures
            if args.save:
                gain = 1/mixture.max()
                dir_name = os.path.basename(test_dir)
                output_dir = os.path.join(model_dir, 'audio', dir_name)
                # TODO: testing a model on two datasets having the same
                # basename but different absolute paths will result in
                # identical output audio directories! Risk of overwriting
                # files unintentionally
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                sf.write(
                    os.path.join(output_dir, f'mixture_enhanced_{k}.wav'),
                    mixture_enhanced*gain,
                    config.PRE.FS,
                )
                # sf.write(
                #     os.path.join(output_dir, f'mixture_ref_{k}.wav'),
                #     mixture_ref*gain,
                #     config.PRE.FS,
                # )
                # sf.write(
                #     os.path.join(output_dir, f'foreground_ref_{k}.wav'),
                #     foreground_ref*gain,
                #     config.PRE.FS,
                # )

            # load oracle signals
            _tag = f'oracle_{next(iter(config.POST.LABELS))}'
            mixture_o = h5f[f'mixture_{_tag}'][k].reshape(-1, 2)
            foreground_o = h5f[f'foreground_{_tag}'][k].reshape(-1, 2)
            background_o = h5f[f'background_{_tag}'][k].reshape(-1, 2)
            noise_o = h5f[f'noise_{_tag}'][k].reshape(-1, 2)
            reverb_o = h5f[f'late_target_{_tag}'][k].reshape(-1, 2)

            # segmental SNRs
            segSSNR, segBR, segNR, segRR = segmental_scores(
                foreground_ref,
                foreground_o,
                background_ref,
                background_o,
                noise_ref,
                noise_o,
                reverb_ref,
                reverb_o,
            )
            scores[test_dir]['oracle']['segSSNR'].append(segSSNR)
            scores[test_dir]['oracle']['segBR'].append(segBR)
            scores[test_dir]['oracle']['segNR'].append(segNR)
            scores[test_dir]['oracle']['segRR'].append(segRR)

            # write oracle enhanced mixture
            if args.save_oracle:
                gain = 1/mixture.max()
                dir_name = os.path.basename(test_dir)
                output_dir = os.path.join(model_dir, 'audio', dir_name)
                # TODO: testing a model on two datasets having the same
                # basename but different absolute paths will result in
                # identical output audio directories! Risk of overwriting
                # files unintentionally
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                sf.write(
                    os.path.join(output_dir, f'mixture_oracle_{k}.wav'),
                    mixture_o*gain,
                    config.PRE.FS,
                )

            # calculate PESQ
            scores[test_dir]['enhanced']['PESQ'].append(pesq(
                config.PRE.FS,
                foreground_ref,
                mixture_enhanced,
                'wb',
            ))
            scores[test_dir]['oracle']['PESQ'].append(pesq(
                config.PRE.FS,
                foreground_ref,
                mixture_o,
                'wb',
            ))
            scores[test_dir]['ref']['PESQ'].append(pesq(
                config.PRE.FS,
                foreground_ref,
                mixture_ref,
                'wb',
            ))

            # calculate STOI
            scores[test_dir]['enhanced']['STOI'].append(stoi(
                foreground_ref,
                mixture_enhanced,
                config.PRE.FS,
            ))
            scores[test_dir]['oracle']['STOI'].append(stoi(
                foreground_ref,
                mixture_o,
                config.PRE.FS,
            ))
            scores[test_dir]['ref']['STOI'].append(stoi(
                foreground_ref,
                mixture_ref,
                config.PRE.FS,
            ))

            # measure time
            enhancement_time += time.time() - start_time
            n_mixtures_enhanced += 1

    # close hdf5 file
    h5f.close()

    # round all floats before saving scores
    def round_dict(d, digits=4):
        for key, val in d.items():
            if isinstance(val, dict):
                d[key] = round_dict(val, digits)
            elif isinstance(val, list):
                d[key] = [round(x, digits) for x in val]
            elif isinstance(val, float):
                d[key] = round(val, digits)
    round_dict(scores)

    # save scores
    with open(os.path.join(), 'r') as f:
        yaml.dump(scores, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('input', nargs='+',
                        help='input model directories')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    parser.add_argument('--no-cuda', action='store_true',
                        help='force testing on cpu')
    parser.add_argument('--save', action='store_true',
                        help='write enhanced signals')
    parser.add_argument('--save-oracle', action='store_true',
                        help='write oracle signals')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    model_dirs = []
    for input_ in args.input:
        if not glob(input_):
            logging.info(f'Model not found: {input_}')
        model_dirs += glob(input_)
    for model_dir in model_dirs:
        main(model_dir, args)
