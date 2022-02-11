import argparse
import logging
import os
import pickle
import time
import random
import json

import h5py
import numpy as np
# from pesq import pesq
from pystoi import stoi
import soundfile as sf
import torch

from brever.config import get_config
from brever.data import DNNDataset, ConvTasNetDataset
from brever.models import DNN, ConvTasNet
from brever.utils import wola
from brever.logger import set_logger


def main():
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

    # disable gradients and set model to eval
    torch.set_grad_enabled(False)
    model.eval()

    # main loop
    for i in range(len(dataset)):

        if config.ARCH == 'dnn':
            data, target = dataset.load_segment(i)
            data, target, mix = dataset.post_proc(data, target, return_mix=True)
            data = data.unsqueeze(0)
            output = model(data)
            output = output.squeeze(0)
            prm = output.numpy().T
            output = wola(prm)[:mix.shape[-1], :].T
            output = output[:, np.newaxis, :]
            mix = mix.numpy()
            output = mix*output
            output = dataset.filterbank.rfilt(output)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(target)
            plt.figure()
            plt.imshow(prm)
            plt.figure()
            plt.plot(output.T)
            plt.figure()
            plt.plot(dataset.filterbank.rfilt(mix).T)
            plt.show()
            import pdb; pdb.set_trace()
            sf.write('temp.wav', output.T, 16000)



    # main loop
    for test_dir in bm.globbed(config.POST.PATH.TEST):
        # check if already tested
        if test_dir in scores.keys() and not args.force:
            logging.info(f'{test_dir} was already tested, skipping')
            continue

        # start clock
        start_time = time.time()

        # verbose and initialize scores field
        logging.info(f'Processing {test_dir}:')
        scores[test_dir] = {
            'model': {
                'MSE': [],
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

        # initialize dataset
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

        # set normalization transform
        if config.POST.NORMALIZATION.TYPE == 'global':
            test_dataset.transform = bdata.TensorStandardizer(mean, std)
        elif config.POST.NORMALIZATION.TYPE == 'recursive':
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

        # open hdf5 file
        h5f = h5py.File(test_dataset.filepath, 'r')

        # loop over mixtures
        n = len(h5f['mixture'])
        start_time_enhancement = time.time()
        for k in range(n):
            if k == 0:
                logging.info(f'Enhancing mixture {k}/{n}...')
            else:
                time_per_mix = (time.time() - start_time_enhancement)/k
                logging.info(f'Enhancing mixture {k}/{n}... '
                             f'Average enhancement time: '
                             f'{time_per_mix:.2f}')

            # load mixture
            mixture = h5f['mixture'][k].reshape(-1, 2)
            # foreground = h5f['foreground'][k].reshape(-1, 2)
            # background = h5f['background'][k].reshape(-1, 2)
            # noise = h5f['noise'][k].reshape(-1, 2)
            # reverb = h5f['late_target'][k].reshape(-1, 2)
            i_start, i_end = test_dataset.file_indices[k]

            # scale signal
            scaler.fit(mixture)
            mixture = scaler.scale(mixture)
            # foreground = scaler.scale(foreground)
            # background = scaler.scale(background)
            # noise = scaler.scale(noise)
            # reverb = scaler.scale(reverb)
            scaler.__init__(scaler.active)

            # apply filterbank
            mixture_filt = filterbank.filt(mixture)
            # foreground_filt = filterbank.filt(foreground)
            # background_filt = filterbank.filt(background)
            # noise_filt = filterbank.filt(noise)
            # reverb_filt = filterbank.filt(reverb)

            # extract features
            features, IRM = test_dataset[i_start:i_end]
            features = torch.from_numpy(features).float()
            IRM = torch.from_numpy(IRM).float()
            if config.MODEL.CUDA and not args.no_cuda:
                features = features.cuda()
                IRM = IRM.cuda()

            # make mask prediction and calculate MSE
            model.eval()
            with torch.no_grad():
                PRM = model(features)
                loss = criterion(PRM, IRM)
                scores[test_dir]['model']['MSE'].append(loss.item())

            # extrapolate predicted mask
            if config.MODEL.CUDA and not args.no_cuda:
                PRM = PRM.cpu()
            PRM = PRM.numpy()
            PRM = wola(PRM, trim=len(mixture_filt))[:, :, np.newaxis]

            # apply predicted mask and reverse filter
            mixture_enhanced = filterbank.rfilt(mixture_filt*PRM)
            # foreground_enhanced = filterbank.rfilt(foreground_filt*PRM)
            # background_enhanced = filterbank.rfilt(background_filt*PRM)
            # noise_enhanced = filterbank.rfilt(noise_filt*PRM)
            # reverb_enhanced = filterbank.rfilt(reverb_filt*PRM)

            # load reference signals
            mixture_ref = h5f['mixture_ref'][k].reshape(-1, 2)
            foreground_ref = h5f['foreground_ref'][k].reshape(-1, 2)
            # background_ref = h5f['background_ref'][k].reshape(-1, 2)
            # noise_ref = h5f['noise_ref'][k].reshape(-1, 2)
            # reverb_ref = h5f['late_target_ref'][k].reshape(-1, 2)

            # # segmental SNRs
            # segSSNR, segBR, segNR, segRR = segmental_scores(
            #     foreground_ref,
            #     foreground_enhanced,
            #     background_ref,
            #     background_enhanced,
            #     noise_ref,
            #     noise_enhanced,
            #     reverb_ref,
            #     reverb_enhanced,
            # )
            # scores[test_dir]['model']['segSSNR'].append(segSSNR)
            # scores[test_dir]['model']['segBR'].append(segBR)
            # scores[test_dir]['model']['segNR'].append(segNR)
            # scores[test_dir]['model']['segRR'].append(segRR)

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

            # # load oracle signals
            # _tag = f'oracle_{next(iter(config.POST.LABELS))}'
            # mixture_o = h5f[f'mixture_{_tag}'][k].reshape(-1, 2)
            # foreground_o = h5f[f'foreground_{_tag}'][k].reshape(-1, 2)
            # background_o = h5f[f'background_{_tag}'][k].reshape(-1, 2)
            # noise_o = h5f[f'noise_{_tag}'][k].reshape(-1, 2)
            # reverb_o = h5f[f'late_target_{_tag}'][k].reshape(-1, 2)

            # # segmental SNRs
            # segSSNR, segBR, segNR, segRR = segmental_scores(
            #     foreground_ref,
            #     foreground_o,
            #     background_ref,
            #     background_o,
            #     noise_ref,
            #     noise_o,
            #     reverb_ref,
            #     reverb_o,
            # )
            # scores[test_dir]['oracle']['segSSNR'].append(segSSNR)
            # scores[test_dir]['oracle']['segBR'].append(segBR)
            # scores[test_dir]['oracle']['segNR'].append(segNR)
            # scores[test_dir]['oracle']['segRR'].append(segRR)

            # # write oracle enhanced mixture
            # if args.save_oracle:
            #     gain = 1/mixture.max()
            #     dir_name = os.path.basename(test_dir)
            #     output_dir = os.path.join(model_dir, 'audio', dir_name)
            #     # TODO: testing a model on two datasets having the same
            #     # basename but different absolute paths will result in
            #     # identical output audio directories! Risk of overwriting
            #     # files unintentionally
            #     if not os.path.exists(output_dir):
            #         os.makedirs(output_dir)
            #     sf.write(
            #         os.path.join(output_dir, f'mixture_oracle_{k}.wav'),
            #         mixture_o*gain,
            #         config.PRE.FS,
            #     )

            # calculate PESQ
            scores[test_dir]['model']['PESQ'].append(pesq(
                config.PRE.FS,
                foreground_ref.mean(axis=1),
                mixture_enhanced.mean(axis=1),
                'wb',
            ))
            # scores[test_dir]['oracle']['PESQ'].append(pesq(
            #     config.PRE.FS,
            #     foreground_ref.mean(axis=1),
            #     mixture_o.mean(axis=1),
            #     'wb',
            # ))
            scores[test_dir]['ref']['PESQ'].append(pesq(
                config.PRE.FS,
                foreground_ref.mean(axis=1),
                mixture_ref.mean(axis=1),
                'wb',
            ))

            # calculate STOI
            scores[test_dir]['model']['STOI'].append(stoi(
                foreground_ref.mean(axis=1),
                mixture_enhanced.mean(axis=1),
                config.PRE.FS,
            ))
            # scores[test_dir]['oracle']['STOI'].append(stoi(
            #     foreground_ref.mean(axis=1),
            #     mixture_o.mean(axis=1),
            #     config.PRE.FS,
            # ))
            scores[test_dir]['ref']['STOI'].append(stoi(
                foreground_ref.mean(axis=1),
                mixture_ref.mean(axis=1),
                config.PRE.FS,
            ))

        # update scores file
        bm.dump_json(scores, scores_path)

        # log time spent
        logging.info(f'Time spent: {time.time() - start_time:.2f}')

        # close hdf5 file
        h5f.close()

    # round and cast to built-in float type before saving scores
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('input',
                        help='model directory')
    parser.add_argument('test_path',
                        help='test dataset path')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    parser.add_argument('--cpu', action='store_true',
                        help='force testing on cpu')
    parser.add_argument('--save', action='store_true',
                        help='write enhanced signals')
    args = parser.parse_args()
    main()
