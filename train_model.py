import os
import argparse
import logging
import time
import pprint
import json
from glob import glob

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from brever.config import defaults
from brever.pytorchtools import (EarlyStopping, TensorStandardizer, H5Dataset,
                                 Feedforward, get_mean_and_std, evaluate)
from brever.modelmanagement import get_feature_indices, get_file_indices


def clear_logger():
    logger = logging.getLogger()
    for i in reversed(range(len(logger.handlers))):
        logger.handlers[i].close()
        logger.removeHandler(logger.handlers[i])


def set_logger(output_dir):
    logger = logging.getLogger()
    logfile = os.path.join(output_dir, 'log.txt')
    filehandler = logging.FileHandler(logfile, mode='w')
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)


def check_overlapping_files(train_path, val_path):
    train_info_path = os.path.join(train_path, 'mixture_info.json')
    val_info_path = os.path.join(val_path, 'mixture_info.json')
    with open(train_info_path, 'r') as f:
        train_info = json.load(f)
    with open(val_info_path, 'r') as f:
        val_info = json.load(f)

    train_targets = [x['target_filename'] for x in train_info]
    val_targets = [x['target_filename'] for x in val_info]
    assert not set(train_targets) & set(val_targets)

    train_noises = [x['directional_noises_filenames'] for x in train_info]
    val_noises = [x['directional_noises_filenames'] for x in val_info]
    train_noises = [x for sublist in train_noises for x in sublist]
    val_noises = [x for sublist in val_noises for x in sublist]
    assert not set(train_noises) & set(val_noises)


def train(model, criterion, optimizer, train_dataloader, cuda):
    model.train()
    for data, target in train_dataloader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def plot_losses(train_losses, val_losses, output_dir):
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')
    fig, ax = plt.subplots()
    ax.plot(train_losses)
    ax.plot(val_losses)
    ax.legend(['training loss', 'validation loss'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    ax.grid(True)
    plot_output_path = os.path.join(output_dir, 'training.png')
    fig.tight_layout()
    fig.savefig(plot_output_path)


def main(model_dir, force):
    logging.info(f'Processing {model_dir}')

    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    # check if model is already trained using directory contents
    train_losses_path = os.path.join(model_dir, 'train_losses.npy')
    val_losses_path = os.path.join(model_dir, 'val_losses.npy')
    if os.path.exists(train_losses_path) and os.path.exists(val_losses_path):
        if not force:
            logging.info(f'Model is already trained.')
            return

    # set logger
    clear_logger()
    set_logger(model_dir)

    # print model info
    logging.info('\n' + pprint.pformat({
        'POST': config.POST.todict(),
        'MODEL': config.MODEL.todict(),
    }))

    # check that there are no overlapping files between train and val sets
    check_overlapping_files(config.POST.PATH.TRAIN, config.POST.PATH.VAL)

    # seed for reproducibility
    torch.manual_seed(0)

    # get features indices from feature extractor instance
    feature_indices = get_feature_indices(config.POST.PATH.TRAIN,
                                          config.POST.FEATURES)

    # get files indices from mixture info file
    file_indices = get_file_indices(config.POST.PATH.TRAIN)

    # load datasets and dataloaders
    logging.info('Loading datasets...')
    train_dataset_path = os.path.join(config.POST.PATH.TRAIN, 'dataset.hdf5')
    val_dataset_path = os.path.join(config.POST.PATH.VAL, 'dataset.hdf5')

    train_dataset = H5Dataset(
        filepath=train_dataset_path,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        decimation=config.POST.DECIMATION,
        feature_indices=feature_indices,
        file_indices=file_indices,
    )
    val_dataset = H5Dataset(
        filepath=val_dataset_path,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        decimation=config.POST.DECIMATION,
        feature_indices=feature_indices,
        file_indices=file_indices,
    )
    logging.info(f'Number of features: {train_dataset.n_features}')

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.MODEL.BATCHSIZE,
        shuffle=config.MODEL.SHUFFLE,
        num_workers=config.MODEL.NWORKERS,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.MODEL.BATCHSIZE,
        shuffle=config.MODEL.SHUFFLE,
        num_workers=config.MODEL.NWORKERS,
        drop_last=True,
    )

    # set normalization transform
    if config.POST.GLOBALSTANDARDIZATION:
        logging.info('Calculating mean and std...')
        mean, std = get_mean_and_std(train_dataloader, config.POST.LOAD)
        train_dataset.transform = TensorStandardizer(mean, std)
        val_dataset.transform = TensorStandardizer(mean, std)

    # initialize network
    model = Feedforward(
        input_size=train_dataset.n_features,
        output_size=train_dataset.n_labels,
        n_layers=config.MODEL.NLAYERS,
        dropout_toggle=config.MODEL.DROPOUT.ON,
        dropout_rate=config.MODEL.DROPOUT.RATE,
        batchnorm_toggle=config.MODEL.BATCHNORM.ON,
        batchnorm_momentum=config.MODEL.BATCHNORM.MOMENTUM,
    )
    if config.MODEL.CUDA:
        model = model.cuda()

    # initialize criterion and optimizer
    criterion = getattr(torch.nn, config.MODEL.CRITERION)()
    optimizer = getattr(torch.optim, config.MODEL.OPTIMIZER)(
        params=model.parameters(),
        lr=config.MODEL.LEARNINGRATE,
        weight_decay=config.MODEL.WEIGHTDECAY,
    )

    # initialize early stopping object
    early_stopping = EarlyStopping(
        patience=config.MODEL.EARLYSTOP.PATIENCE,
        verbose=config.MODEL.EARLYSTOP.VERBOSE,
        delta=config.MODEL.EARLYSTOP.DELTA,
        checkpoint_dir=model_dir,
    )

    # main loop
    logging.info('Starting main loop...')
    train_losses = []
    val_losses = []
    start_time = time.time()
    for epoch in range(config.MODEL.EPOCHS):
        # train
        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            cuda=config.MODEL.CUDA,
        )

        # validate
        train_loss = evaluate(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            load=config.POST.LOAD,
            cuda=config.MODEL.CUDA,
        )
        val_loss = evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            load=config.POST.LOAD,
            cuda=config.MODEL.CUDA,
        )

        # log and store errors
        logging.info(f'Epoch {epoch}: train loss: {train_loss:.6f}; '
                     f'val loss: {val_loss:.6f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info('Early stopping!')
            logging.info(f'Best validation loss: '
                         f'{early_stopping.val_loss_min}')
            break

    # display total time spent
    total_time = time.time() - start_time
    logging.info(f'Time spent: {int(total_time/3600)} h '
                 f'{int(total_time%3600/60)} m {int(total_time%60)} s')

    # plot training and validation error
    plot_losses(train_losses, val_losses, model_dir)

    # save errors
    np.save(train_losses_path, train_losses)
    np.save(val_losses_path, val_losses)

    # close log file handler and rename model
    clear_logger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('-i', '--input',
                        help=('Input model directory.'))
    parser.add_argument('-f', '--force',
                        help=('Train even if already trained.'),
                        action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
    )

    for model_dir in glob(args.input):
        main(model_dir, args.force)
