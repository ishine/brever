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
import brever.pytorchtools as bptt
from brever.modelmanagement import globbed


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
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
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
    if set(train_targets) & set(val_targets):
        logging.warning('Training and validation speech materials are '
                        'overlapping')

    train_noises = [y['filename'] for x in train_info
                    for y in x['directional']['sources']]
    val_noises = [y['filename'] for x in val_info
                  for y in x['directional']['sources']]
    if set(train_noises) & set(val_noises):
        logging.warning('Training and validation noise materials are '
                        'overlapping')


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


def plot_losses(train_losses, val_losses, test_losses, output_dir):
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')
    fig, ax = plt.subplots()
    ax.plot(train_losses)
    ax.plot(val_losses)
    ax.plot(test_losses)
    ax.legend(['training loss', 'validation loss', 'test loss'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    ax.grid(True)
    plot_output_path = os.path.join(output_dir, 'training.png')
    fig.tight_layout()
    fig.savefig(plot_output_path)
    plt.close(fig)


def main(model_dir, force, no_cuda):
    logging.info(f'Processing {model_dir}')

    # load config file
    config = defaults()
    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        config.update(yaml.safe_load(f))

    # check if model is already trained using directory contents
    loss_path = os.path.join(model_dir, 'losses.npz')
    if os.path.exists(loss_path):
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
    logging.info('Checking if any material files are overlapping')
    check_overlapping_files(config.POST.PATH.TRAIN, config.POST.PATH.VAL)

    # seed for reproducibility
    torch.manual_seed(config.MODEL.SEED)

    # initialize datasets
    logging.info('Initializing training dataset')
    train_dataset = bptt.H5Dataset(
        dirpath=config.POST.PATH.TRAIN,
        features=config.POST.FEATURES,
        labels=config.POST.LABELS,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        decimation=config.POST.DECIMATION,
        dct_toggle=config.POST.DCT.ON,
        n_dct=config.POST.DCT.NCOEFF,
        file_based_stats=config.POST.STANDARDIZATION.FILEBASED,
        prestack=config.POST.PRESTACK,
    )
    logging.info('Initializing validation dataset')
    val_dataset = bptt.H5Dataset(
        dirpath=config.POST.PATH.VAL,
        features=config.POST.FEATURES,
        labels=config.POST.LABELS,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        decimation=config.POST.DECIMATION,
        dct_toggle=config.POST.DCT.ON,
        n_dct=config.POST.DCT.NCOEFF,
        file_based_stats=config.POST.STANDARDIZATION.FILEBASED,
        prestack=config.POST.PRESTACK,
    )
    logging.info(f'Number of features: {train_dataset.n_features}')

    # initialize dataloaders
    logging.info('Initializing training dataloader')
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.MODEL.BATCHSIZE,
        shuffle=config.MODEL.SHUFFLE,
        num_workers=config.MODEL.NWORKERS,
        drop_last=True,
    )
    logging.info('Initializing validation dataloader')
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.MODEL.BATCHSIZE,
        shuffle=config.MODEL.SHUFFLE,
        num_workers=config.MODEL.NWORKERS,
        drop_last=True,
    )

    # set normalization transform
    logging.info('Calculating mean and std')
    if config.POST.STANDARDIZATION.FILEBASED:
        train_means, train_stds = bptt.get_files_mean_and_std(
            train_dataset,
            config.POST.STANDARDIZATION.UNIFORMFEATURES,
        )
        train_dataset.transform = bptt.StateTensorStandardizer(
            train_means,
            train_stds,
        )
        val_means, val_stds = bptt.get_files_mean_and_std(
            val_dataset,
            config.POST.STANDARDIZATION.UNIFORMFEATURES,
        )
        val_dataset.transform = bptt.StateTensorStandardizer(
            val_means,
            val_stds,
        )
    else:
        mean, std = bptt.get_mean_and_std(
            train_dataset,
            train_dataloader,
            config.POST.STANDARDIZATION.UNIFORMFEATURES,
        )
        to_save = np.vstack((mean, std))
        stat_path = os.path.join(model_dir, 'statistics.npy')
        np.save(stat_path, to_save)
        train_dataset.transform = bptt.TensorStandardizer(mean, std)
        val_dataset.transform = bptt.TensorStandardizer(mean, std)

    # initialize test dataset and dataloaders
    test_dataloaders = []
    for test_dir in globbed(config.POST.PATH.TEST):
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

        # add to list of dataloaders
        test_dataloaders.append(test_dataloader)

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
        'hidden_sizes': config.MODEL.HIDDENSIZES,
    }
    model = bptt.Feedforward(**model_args)
    model_args_path = os.path.join(model_dir, 'model_args.yaml')
    with open(model_args_path, 'w') as f:
        yaml.dump(model_args, f)
    if config.MODEL.CUDA and not no_cuda:
        model = model.cuda()

    # initialize criterion and optimizer
    criterion = getattr(torch.nn, config.MODEL.CRITERION)()
    optimizer = getattr(torch.optim, config.MODEL.OPTIMIZER)(
        params=model.parameters(),
        lr=config.MODEL.LEARNINGRATE,
        weight_decay=config.MODEL.WEIGHTDECAY,
    )

    # initialize early stopper
    earlyStop = bptt.EarlyStopping(
        patience=config.MODEL.EARLYSTOP.PATIENCE,
        verbose=config.MODEL.EARLYSTOP.VERBOSE,
        delta=config.MODEL.EARLYSTOP.DELTA,
    )

    # initialize progress tracker
    progressTracker = bptt.ProgressTracker(
        strip=config.MODEL.PROGRESS.STRIP,
        threshold=config.MODEL.PROGRESS.THRESHOLD,
    )

    # main loop
    logging.info('Starting main loop')
    train_losses = []
    val_losses = []
    test_losses = []
    total_time = 0
    start_time = time.time()
    for epoch in range(config.MODEL.EPOCHS):
        # train
        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            cuda=config.MODEL.CUDA and not no_cuda,
        )

        # evaluate
        train_loss = bptt.evaluate(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            cuda=config.MODEL.CUDA and not no_cuda,
        )
        val_loss = bptt.evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            cuda=config.MODEL.CUDA and not no_cuda,
        )
        test_loss = np.mean([
            bptt.evaluate(
                model=model,
                criterion=criterion,
                dataloader=test_dataloader,
                cuda=config.MODEL.CUDA and not no_cuda,
            )
            for test_dataloader in test_dataloaders])

        # log and store errors
        total_time = time.time() - start_time
        time_per_epoch = total_time/(epoch+1)
        logging.info(f'Epoch {epoch}: train loss: {train_loss:.6f}; '
                     f'val loss: {val_loss:.6f}; '
                     f'Time per epoch: {time_per_epoch:.2f} s')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        # check stop criterion
        checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
        if config.MODEL.EARLYSTOP.ON:
            earlyStop(val_loss, model, checkpoint_path)
            if earlyStop.stop:
                logging.info('Early stopping!')
                logging.info(f'Best validation loss: {earlyStop.min_loss}')
                break
        elif config.MODEL.PROGRESS.ON:
            progressTracker(train_loss, model, checkpoint_path)
            if progressTracker.stop:
                logging.info('Train loss has converged')
                break
        else:
            raise ValueError("Can't have both early stopping and progress "
                             "criterion")

    # display total time spent
    total_time = time.time() - start_time
    logging.info(f'Time spent: {int(total_time/3600)} h '
                 f'{int(total_time%3600/60)} m {int(total_time%60)} s')

    # plot training and validation error
    plot_losses(train_losses, val_losses, test_losses, model_dir)

    # save errors
    np.savez(loss_path, train=train_losses, val=val_losses, test=test_losses)

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
    parser.add_argument('--no-cuda', action='store_true',
                        help='force training on cpu')
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
        main(model_dir, args.force, args.no_cuda)
