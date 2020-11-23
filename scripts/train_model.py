import os
import argparse
import logging
import time
import json
from glob import glob
import sys

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from brever.config import defaults
from brever.pytorchtools import (EarlyStopping, TensorStandardizer, H5Dataset,
                                 Feedforward, get_mean_and_std, evaluate)


def clear_logger():
    logger = logging.getLogger()
    for i in reversed(range(len(logger.handlers))):
        logger.handlers[i].close()
        logger.removeHandler(logger.handlers[i])


def set_logger(output_dir):
    logger = logging.getLogger()
    logfile = os.path.join(output_dir, 'log.txt')
    filehandler = logging.FileHandler(logfile, mode='w')
    streamhandler = logging.StreamHandler(sys.stdout)
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

    train_targets = [x['target']['filename'] for x in train_info]
    val_targets = [x['target']['filename'] for x in val_info]
    assert not set(train_targets) & set(val_targets)

    train_noises = [y['filename'] for x in train_info
                    for y in x['directional']['sources']]
    val_noises = [y['filename'] for x in val_info
                  for y in x['directional']['sources']]
    assert not set(train_noises) & set(val_noises)


def train(model, criterion, optimizer, train_dataloader, cuda):
    model.train()
    for data, target in train_dataloader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = data.float(), target.float()
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

    # load config file
    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    # check if model is already trained using directory contents
    train_loss_path = os.path.join(model_dir, 'train_losses.npy')
    val_loss_path = os.path.join(model_dir, 'val_losses.npy')
    if os.path.exists(train_loss_path) and os.path.exists(val_loss_path):
        if not force:
            logging.info('Model is already trained')
            return

    # set logger
    clear_logger()
    set_logger(model_dir)

    # print model info
    logging.info(yaml.dump({
        'POST': config.POST.to_dict(),
        'MODEL': config.MODEL.to_dict(),
    }))

    # check that there are no overlapping files between train and val sets
    check_overlapping_files(config.POST.PATH.TRAIN, config.POST.PATH.VAL)

    # seed for reproducibility
    torch.manual_seed(0)

    # initialize datasets
    train_dataset = H5Dataset(
        dirpath=config.POST.PATH.TRAIN,
        features=config.POST.FEATURES,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        decimation=config.POST.DECIMATION,
        dct=config.POST.DCT.ON,
        n_dct=config.POST.DCT.NCOEFF,
    )
    val_dataset = H5Dataset(
        dirpath=config.POST.PATH.VAL,
        features=config.POST.FEATURES,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        decimation=config.POST.DECIMATION,
        dct=config.POST.DCT.ON,
        n_dct=config.POST.DCT.NCOEFF,
    )
    logging.info(f'Number of features: {train_dataset.n_features}')

    # initialize dataloaders
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
    if config.POST.STANDARDIZATION.FILEBASED:
        pass
    else:
        logging.info('Calculating mean and std')
        mean, std = get_mean_and_std(
            train_dataset,
            train_dataloader,
            config.POST.STANDARDIZATION.UNIFORMFEATURES,
        )
        to_save = np.vstack((mean, std))
        stat_path = os.path.join(model_dir, 'statistics.npy')
        np.save(stat_path, to_save)
        train_dataset.transform = TensorStandardizer(mean, std)
        val_dataset.transform = TensorStandardizer(mean, std)

    # initialize network
    model_args = {
        'input_size': train_dataset.n_features,
        'output_size': train_dataset.n_labels,
        'n_layers': config.MODEL.NLAYERS,
        'dropout_toggle': config.MODEL.DROPOUT.ON,
        'dropout_rate': config.MODEL.DROPOUT.RATE,
        'dropout_input': config.MODEL.DROPOUT.INPUT,
        'batchnorm_toggle': config.MODEL.BATCHNORM.ON,
        'batchnorm_momentum': config.MODEL.BATCHNORM.MOMENTUM,
    }
    model = Feedforward(**model_args)
    model_args_path = os.path.join(model_dir, 'model_args.yaml')
    with open(model_args_path, 'w') as f:
        yaml.dump(model_args, f)
    if config.MODEL.CUDA:
        model = model.cuda()

    # initialize criterion and optimizer
    criterion = getattr(torch.nn, config.MODEL.CRITERION)()
    optimizer = getattr(torch.optim, config.MODEL.OPTIMIZER)(
        params=model.parameters(),
        lr=config.MODEL.LEARNINGRATE,
        weight_decay=config.MODEL.WEIGHTDECAY,
    )

    # initialize early stopper
    early_stopping = EarlyStopping(
        patience=config.MODEL.EARLYSTOP.PATIENCE,
        verbose=config.MODEL.EARLYSTOP.VERBOSE,
        delta=config.MODEL.EARLYSTOP.DELTA,
        checkpoint_dir=model_dir,
    )

    # main loop
    logging.info('Starting main loop')
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

        # evaluate
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
    np.save(train_loss_path, train_losses)
    np.save(val_loss_path, val_losses)

    # write full config file
    full_config_file = os.path.join(model_dir, 'config_full.yaml')
    with open(full_config_file, 'w') as f:
        yaml.dump(config.to_dict(), f)

    # close log file handler
    clear_logger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a model')
    parser.add_argument('input', nargs='+',
                        help='input model directories')
    parser.add_argument('-f', '--force', action='store_true',
                        help='train even if already trained')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
    )

    model_dirs = []
    for input_ in args.input:
        if not glob(input_):
            logging.info(f'Model not found: {input_}')
        model_dirs += glob(input_)
    for model_dir in model_dirs:
        main(model_dir, args.force)
