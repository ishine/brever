import argparse
import logging
import os
import random

import numpy as np
import torch

from brever.config import get_config
from brever.data import BreverDataset
from brever.logger import set_logger
from brever.models import initialize_model, count_params
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

    # initialize model
    logging.info('Initializing model')
    model = initialize_model(config)

    # initialize dataset
    logging.info('Initializing dataset')
    kwargs = {}
    if hasattr(config.MODEL, 'SOURCES'):
        kwargs['components'] = config.MODEL.SOURCES
    dataset = BreverDataset(
        path=config.TRAINING.PATH,
        segment_length=config.TRAINING.SEGMENT_LENGTH,
        fs=config.FS,
        model=model,
        **kwargs,
    )

    # preload data
    if config.TRAINING.PRELOAD:
        logging.info('Preloading data')
        dataset.preload(cuda)

    # train val split
    val_length = int(len(dataset)*config.TRAINING.VAL_SIZE)
    train_length = len(dataset) - val_length
    train_split, val_split = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )

    # set normalization statistics
    if hasattr(config.MODEL, 'NORMALIZATION'):
        if config.MODEL.NORMALIZATION.TYPE == 'static':
            logging.info('Calculating training statistics')
            mean, std = train_split.dataset.get_statistics()
            model.normalization.set_statistics(mean, std)
        else:
            raise ValueError('Unrecognized normalization type, got '
                             f'{config.MODEL.NORMALIZATION.TYPE}')

    # cast to cuda
    if cuda:
        model = model.cuda()

    # print number of parameters
    num_params = f'{round(count_params(model))/1e6:.2f}M'
    logging.info(f'Number of parameters: {num_params}')

    # initialize trainer
    logging.info('Initializing trainer')
    trainer = BreverTrainer(
        model=model,
        train_dataset=train_split,
        val_dataset=val_split,
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
        ignore_checkpoint=args.force,
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
