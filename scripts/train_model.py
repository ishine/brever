import argparse
import logging
import os
import random

import numpy as np
import torch

from brever.config import get_config
from brever.data import DNNDataset, ConvTasNetDataset
from brever.logger import set_logger
from brever.models import DNN, ConvTasNet
from brever.training import BreverTrainer


def main():
    # check if already trained
    loss_path = os.path.join(args.input, 'losses.npz')
    if os.path.exists(loss_path) and not args.force:
        raise FileExistsError(f'training already done: {loss_path}')

    # load model config
    config_path = os.path.join(args.input, 'config.yaml')
    config = get_config(config_path)
    cuda = config.TRAINING.CUDA and not args.cpu

    # initialize logger
    log_file = os.path.join(args.input, 'log.log')
    set_logger(log_file)
    logging.info(f'Training {args.input}')
    logging.info(config.to_dict())

    # seed for reproducibility
    random.seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)
    torch.manual_seed(config.TRAINING.SEED)

    # initialize dataset
    logging.info('Initializing dataset')
    if config.ARCH == 'dnn':
        dataset = DNNDataset(
            path=config.TRAINING.PATH,
            segment_length=config.TRAINING.SEGMENT_LENGTH,
            fs=config.FS,
            features=config.MODEL.FEATURES,
            stacks=config.MODEL.STACKS,
            decimation=config.MODEL.DECIMATION,
            stft_frame_length=config.MODEL.STFT.FRAME_LENGTH,
            stft_hop_length=config.MODEL.STFT.HOP_LENGTH,
            stft_window=config.MODEL.STFT.WINDOW,
            mel_filters=config.MODEL.MEL_FILTERS,
        )
    elif config.ARCH == 'convtasnet':
        dataset = ConvTasNetDataset(
            path=config.TRAINING.PATH,
            segment_length=config.TRAINING.SEGMENT_LENGTH,
            fs=config.FS,
            components=config.MODEL.SOURCES,
        )
    else:
        raise ValueError(f'wrong model architecture, got {config.ARCH}')

    # preload data
    if config.TRAINING.PRELOAD:
        logging.info('Preloading data')
        dataset.preload(cuda)

    # train val split
    val_length = int(len(dataset)*config.TRAINING.VAL_SIZE)
    train_length = len(dataset) - val_length
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )

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
        if config.MODEL.NORMALIZATION.TYPE == 'static':
            logging.info('Calculating training statistics')
            mean, std = train_dataset.dataset.get_statistics()
            model.normalization.set_statistics(mean, std)
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

    # cast to cuda
    if cuda:
        model = model.cuda()

    # initialize trainer
    logging.info('Initializing trainer')
    trainer = BreverTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dirpath=args.input,
        workers=config.TRAINING.WORKERS,
        epochs=config.TRAINING.EPOCHS,
        learning_rate=config.TRAINING.LEARNING_RATE,
        weight_decay=config.TRAINING.WEIGHT_DECAY,
        cuda=cuda,
        criterion=config.TRAINING.CRITERION,
        optimizer=config.TRAINING.OPTIMIZER,
        batch_sampler=config.TRAINING.BATCH_SAMPLER.WHICH,
        batch_size=config.TRAINING.BATCH_SAMPLER.BATCH_SIZE,
        num_buckets=config.TRAINING.BATCH_SAMPLER.NUM_BUCKETS,
        sorted_=config.TRAINING.BATCH_SAMPLER.SORTED,
        segment_length=config.TRAINING.SEGMENT_LENGTH,
        fs=config.FS,
        early_stop=config.TRAINING.EARLY_STOP.TOGGLE,
        early_stop_patience=config.TRAINING.EARLY_STOP.PATIENCE,
        convergence=config.TRAINING.CONVERGENCE.TOGGLE,
        convergence_window=config.TRAINING.CONVERGENCE.WINDOW,
        convergence_threshold=config.TRAINING.CONVERGENCE.THRESHOLD,
        grad_clip=config.TRAINING.GRAD_CLIP,
    )

    # run
    logging.info('Starting training loop')
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a model')
    parser.add_argument('input',
                        help='model directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='train even if already trained')
    parser.add_argument('--cpu', action='store_true',
                        help='force trainig on cpu')
    args = parser.parse_args()
    main()
