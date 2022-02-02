import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from brever.config import get_config
from brever.data import DNNDataset, BreverBatchSampler, BreverDataLoader
from brever.logger import set_logger
from brever.models import DNN
from brever.training import EarlyStopping, ConvergenceTracker, evaluate


def train(model, criterion, optimizer, dataloader, cuda):
    model.train()
    for data, target in dataloader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = data.float(), target.float()
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
    plt.close(fig)


def main():
    # check if already trained
    loss_path = os.path.join(args.input, 'losses.npz')
    if os.path.exists(loss_path) and not not args.force:
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
    torch.manual_seed(config.TRAINING.SEED)

    # initialize dataset
    logging.info('Initializing dataset')
    if config.MODEL.ARCH == 'dnn':
        dataset = DNNDataset(
            path=config.TRAINING.PATH,
            features=config.MODEL.FEATURES,
        )
    else:
        raise ValueError(f'wrong model architecture, got {config.MODEL.ARCH}')

    # train val split
    val_length = int(len(dataset)*config.TRAINING.VAL_SPLIT)
    train_length = len(dataset) - val_length
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )

    # initialize batch samplers
    logging.info('Initializing batch samplers')
    train_batch_sampler = BreverBatchSampler(
        dataset=train_dataset,
        batch_size=config.TRAINING.BATCH_SIZE,
    )
    val_batch_sampler = BreverBatchSampler(
        dataset=val_dataset,
        batch_size=config.TRAINING.BATCH_SIZE,
    )

    # initialize dataloaders
    logging.info('Initializing dataloaders')
    train_dataloader = BreverDataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.TRAINING.WORKERS,
    )
    val_dataloader = BreverDataLoader(
        dataset=val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=config.TRAINING.WORKERS,
    )

    # initialize model
    logging.info('Initializing model')
    if config.MODEL.ARCH == 'dnn':
        model = DNN(
            input_size=dataset.n_features,
            output_size=dataset.n_labels,
            hidden_layers=config.MODEL.HIDDEN_LAYERS,
            dropout=config.MODEL.DROPOUT,
            batchnorm=config.MODEL.BATCH_NORM.TOGGLE,
            batchnorm_momentum=config.MODEL.BATCH_NORM.MOMENTUM,
        )
    else:
        raise ValueError(f'wrong model architecture, got {config.MODEL.ARCH}')

    # cast to cuda
    if cuda:
        model = model.cuda()

    # initialize criterion and optimizer
    criterion = getattr(torch.nn, config.TRAINING.CRITERION)()
    optimizer = getattr(torch.optim, config.TRAINING.OPTIMIZER)(
        params=model.parameters(),
        lr=config.TRAINING.LEARNING_RATE,
        weight_decay=config.TRAINING.WEIGHT_DECAY,
    )

    # initialize early stopper
    earlyStop = EarlyStopping(
        patience=config.TRAINING.EARLY_STOP.PATIENCE,
    )

    # initialize convergence tracker
    convergenceTracker = ConvergenceTracker(
        window=config.TRAINING.CONVERGENCE.WINDOW,
        threshold=config.TRAINING.CONVERGENCE.THRESHOLD,
    )

    # training loop initialization
    logging.info('Starting training loop')
    checkpoint_path = os.path.join(args.input, 'checkpoint.pt')
    train_losses = []
    val_losses = []
    total_time = 0
    start_time = time.time()

    for epoch in range(config.TRAINING.EPOCHS):

        # evaluate
        train_loss = evaluate(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            cuda=cuda,
        )
        val_loss = evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            cuda=cuda,
        )

        # log and store errors
        if epoch == 0:
            logging.info(f'Epoch {epoch}: train loss: {train_loss:.6f}; '
                         f'val loss: {val_loss:.6f}')
        else:
            time_per_epoch = total_time/epoch
            logging.info(f'Epoch {epoch}: train loss: {train_loss:.6f}; '
                         f'val loss: {val_loss:.6f}; '
                         f'Time per epoch: {time_per_epoch:.2f} s')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # train
        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_dataloader,
            cuda=cuda,
        )

        # check stop criterion
        if config.TRAINING.EARLY_STOP.TOGGLE:
            earlyStop(val_loss, model, checkpoint_path)
            if earlyStop.stop:
                logging.info('Early stopping now')
                logging.info(f'Best validation loss: {earlyStop.min_loss}')
                break
        elif config.TRAINING.CONVERGENCE.TOGGLE:
            convergenceTracker(train_loss, model, checkpoint_path)
            if convergenceTracker.stop:
                logging.info('Train loss has now converged')
                break
        else:
            raise ValueError('cannot have both early stopping and progress '
                             'criterion')

        # update time spent
        total_time = time.time() - start_time

    # display total time spent
    total_time = time.time() - start_time
    logging.info(f'Time spent: {int(total_time/3600)} h '
                 f'{int(total_time%3600/60)} m {int(total_time%60)} s')

    # plot training and validation error
    plot_losses(train_losses, val_losses, args.input)

    # save errors
    np.savez(loss_path, train=train_losses, val=val_losses)


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
