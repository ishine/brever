import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from brever.args import TrainingArgParser
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
    # initialize training config
    train_cfg = get_config('config/training.yaml')
    train_cfg.update_from_args(args, parser.arg_map)
    train_id = train_cfg.get_hash()

    # create training directory
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'{args.input} does not exist')
    train_dir = os.path.join(args.input, 'trainings', train_id)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # check if training already exists
    train_cfg_path = os.path.join(train_dir, 'config.yaml')
    if os.path.exists(train_cfg_path) and not not args.force:
        raise FileExistsError(f'training already done: {train_cfg_path}')

    # load model config
    model_cfg_path = os.path.join(args.input, 'config.yaml')
    model_cfg = get_config(model_cfg_path)

    # initialize logger
    log_file = os.path.join(train_dir, 'log.log')
    set_logger(log_file)
    logging.info(f'Training {args.input}')
    logging.info(model_cfg.to_dict())
    logging.info(train_cfg.to_dict())

    # seed for reproducibility
    torch.manual_seed(train_cfg.SEED)

    # initialize dataset
    logging.info('Initializing dataset')
    if model_cfg.ARCH == 'dnn':
        dataset = DNNDataset(
            path=train_cfg.PATH,
            features=model_cfg.FEATURES,
        )
    else:
        raise ValueError(f'wrong model architecture, got {model_cfg.ARCH}')

    # train val split
    val_length = int(len(dataset)*train_cfg.VAL_SPLIT)
    train_length = len(dataset) - val_length
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )

    # initialize batch samplers
    logging.info('Initializing batch samplers')
    train_batch_sampler = BreverBatchSampler(
        dataset=train_dataset,
        batch_size=train_cfg.BATCH_SIZE,
    )
    val_batch_sampler = BreverBatchSampler(
        dataset=val_dataset,
        batch_size=train_cfg.BATCH_SIZE,
    )

    # initialize dataloaders
    logging.info('Initializing dataloaders')
    train_dataloader = BreverDataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=train_cfg.WORKERS,
    )
    val_dataloader = BreverDataLoader(
        dataset=val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=train_cfg.WORKERS,
    )

    # initialize model
    logging.info('Initializing model')
    if model_cfg.ARCH == 'dnn':
        model = DNN(
            input_size=dataset.n_features,
            output_size=dataset.n_labels,
            hidden_layers=model_cfg.HIDDEN_LAYERS,
            dropout=model_cfg.DROPOUT,
            batchnorm=model_cfg.BATCH_NORM.TOGGLE,
            batchnorm_momentum=model_cfg.BATCH_NORM.MOMENTUM,
        )
    else:
        raise ValueError(f'wrong model architecture, got {model_cfg.ARCH}')

    # cast to cuda
    if train_cfg.CUDA:
        model = model.cuda()

    # initialize criterion and optimizer
    criterion = getattr(torch.nn, train_cfg.CRITERION)()
    optimizer = getattr(torch.optim, train_cfg.OPTIMIZER)(
        params=model.parameters(),
        lr=train_cfg.LEARNING_RATE,
        weight_decay=train_cfg.WEIGHT_DECAY,
    )

    # initialize early stopper
    earlyStop = EarlyStopping(
        patience=train_cfg.EARLY_STOP.PATIENCE,
    )

    # initialize convergence tracker
    convergenceTracker = ConvergenceTracker(
        window=train_cfg.CONVERGENCE.WINDOW,
        threshold=train_cfg.CONVERGENCE.THRESHOLD,
    )

    # training loop initialization
    logging.info('Starting training loop')
    checkpoint_path = os.path.join(train_dir, 'checkpoint.pt')
    train_losses = []
    val_losses = []
    total_time = 0
    start_time = time.time()

    for epoch in range(train_cfg.EPOCHS):

        # evaluate
        train_loss = evaluate(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            cuda=train_cfg.CUDA,
        )
        val_loss = evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            cuda=train_cfg.CUDA,
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
            cuda=train_cfg.CUDA,
        )

        # check stop criterion
        if train_cfg.EARLY_STOP.TOGGLE:
            earlyStop(val_loss, model, checkpoint_path)
            if earlyStop.stop:
                logging.info('Early stopping now')
                logging.info(f'Best validation loss: {earlyStop.min_loss}')
                break
        elif train_cfg.CONVERGENCE.TOGGLE:
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
    plot_losses(train_losses, val_losses, train_dir)

    # save errors
    loss_path = os.path.join(train_dir, 'losses.npz')
    np.savez(loss_path, train=train_losses, val=val_losses)

    # write config file
    with open(train_cfg_path, 'w') as f:
        yaml.dump(train_cfg.to_dict(), f)


if __name__ == '__main__':
    parser = TrainingArgParser(description='train a model')
    parser.add_argument('input',
                        help='model directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='train even if already trained')
    args = parser.parse_args()
    main()
