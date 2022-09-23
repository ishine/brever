import argparse
import logging
import os
import random
import tempfile
import shutil

import numpy as np
import torch

from brever.config import get_config
from brever.data import BreverDataset
from brever.logger import set_logger
from brever.models import initialize_model, count_params
from brever.training import BreverTrainer


def main():
    # create a temporary directory and copy model
    tempdir = tempfile.TemporaryDirectory()
    shutil.copytree(args.input, tempdir.name, dirs_exist_ok=True)

    # load model config
    config_path = os.path.join(tempdir.name, 'config.yaml')
    config = get_config(config_path)
    cuda = config.TRAINING.CUDA and not args.cpu

    # initialize logger
    log_file = os.path.join(tempdir.name, 'log.log')
    set_logger(log_file)
    logging.info(f'Training {tempdir.name}')
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
    if config.TRAINING.BATCH_SAMPLER.DYNAMIC:
        kwargs['dynamic_batch_size'] = config.TRAINING.BATCH_SAMPLER.BATCH_SIZE
    dataset = BreverDataset(
        path=config.TRAINING.PATH,
        segment_length=config.TRAINING.SEGMENT_LENGTH,
        fs=config.FS,
        model=model,
        **kwargs,
    )

    # preload data
    if config.TRAINING.PRELOAD and not args.no_preload:
        logging.info('Preloading data')
        dataset.preload(cuda)

    # train val split
    val_length = int(len(dataset)*config.TRAINING.VAL_SIZE)
    train_length = len(dataset) - val_length
    train_split, val_split = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )

    # set normalization statistics
    if hasattr(config.MODEL, 'NORMALIZATION') and not args.no_norm:
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
        dirpath=tempdir.name,
        workers=0,
        epochs=args.epochs,
        learning_rate=config.TRAINING.LEARNING_RATE,
        weight_decay=config.TRAINING.WEIGHT_DECAY,
        cuda=cuda,
        criterion=config.TRAINING.CRITERION,
        optimizer=config.TRAINING.OPTIMIZER,
        batch_sampler=config.TRAINING.BATCH_SAMPLER.WHICH,
        batch_size=config.TRAINING.BATCH_SAMPLER.BATCH_SIZE,
        num_buckets=config.TRAINING.BATCH_SAMPLER.NUM_BUCKETS,
        dynamic_batch_size=config.TRAINING.BATCH_SAMPLER.DYNAMIC,
        fs=config.FS,
        early_stop=config.TRAINING.EARLY_STOP.TOGGLE,
        early_stop_patience=config.TRAINING.EARLY_STOP.PATIENCE,
        convergence=config.TRAINING.CONVERGENCE.TOGGLE,
        convergence_window=config.TRAINING.CONVERGENCE.WINDOW,
        convergence_threshold=config.TRAINING.CONVERGENCE.THRESHOLD,
        grad_clip=config.TRAINING.GRAD_CLIP,
        ignore_checkpoint=True,
    )

    # run
    logging.info('Starting training loop')
    trainer.run()

    # close temporary directory
    tempdir.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a model')
    parser.add_argument('input',
                        help='model directory')
    parser.add_argument('--cpu', action='store_true',
                        help='force training on cpu, ignoring config file')
    parser.add_argument('--no-preload', action='store_true',
                        help='force no preload ')
    parser.add_argument('--no-norm', action='store_true',
                        help='force no feature normalization')
    parser.add_argument('--epochs', type=int, required=True,
                        help='number of epochs, ignoring config file')
    args = parser.parse_args()
    main()
