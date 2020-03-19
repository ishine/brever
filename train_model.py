import os
import logging
import time
import argparse
import pprint
import json
import pickle
import hashlib

import yaml
import matplotlib.pyplot as plt
import torch

from brever.config import defaults
from brever.pytorchtools import (EarlyStopping, TensorStandardizer, H5Dataset,
                                 Feedforward)


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


def get_unique_id(data):
    if not data:
        data = {}
    unique_str = ''.join([f'{hashlib.sha256(str(key).encode()).hexdigest()}'
                          f'{hashlib.sha256(str(val).encode()).hexdigest()}'
                          for key, val in sorted(data.items())])
    unique_id = hashlib.sha256(unique_str.encode()).hexdigest()
    return unique_id


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


def get_feature_indices(train_path, features):
    pipes_path = os.path.join(train_path, 'pipes.pkl')
    with open(pipes_path, 'rb') as f:
        featureExtractor = pickle.load(f)['featureExtractor']
    names = featureExtractor.features
    indices = featureExtractor.indices
    indices_dict = {name: lims for name, lims in zip(names, indices)}
    feature_indices = [indices_dict[feature] for feature in features]
    return feature_indices


def get_file_indices(train_path):
    metadatas_path = os.path.join(train_path, 'mixture_info.json')
    with open(metadatas_path, 'r') as f:
        metadatas = json.load(metadatas_path)
        indices = [item['dataset_indices'] for item in metadatas]
    return indices


def get_mean_and_std(dataloader, load):
    if load:
        mean = dataloader.dataset[:][0].mean(0)
        std = dataloader.dataset[:][0].std(0)
    else:
        mean = 0
        for data, _ in dataloader:
            mean += data.mean(0)
        mean /= len(dataloader)
        var = 0
        for data, _ in dataloader:
            var += ((data - mean)**2).mean(0)
        var /= len(dataloader)
        std = var**0.5
    return mean, std


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


def eval(model, criterion, train_dataloader, val_dataloader, load, cuda):
    model.eval()
    with torch.no_grad():
        if load:
            data, target = train_dataloader.dataset[:]
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            train_loss = loss.item()

            data, target = val_dataloader.dataset[:]
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            val_loss = loss.item()
        else:
            train_loss = 0
            for data, target in train_dataloader:
                if cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                train_loss += loss.item()
            train_loss /= len(train_dataloader)

            val_loss = 0
            for data, target in val_dataloader:
                if cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
            val_loss /= len(val_dataloader)
    return train_loss, val_loss


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


def main(input_config, force):
    with open(input_config, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)
    output_dir = os.path.dirname(input_config)

    # redirect logger
    clear_logger()
    set_logger(output_dir)

    # rename model
    unique_id = get_unique_id(data)
    output_root = os.path.dirname(output_dir)
    new_output_dir = os.path.join(output_root, unique_id)
    if not force and os.path.exists(new_output_dir):
        logging.info(f'Model {unique_id} already exists.')
        clear_logger()
        return

    # display model info
    logging.info('\n' + pprint.pformat({'POST': config.POST.todict()}))
    logging.info('\n' + pprint.pformat({'MODEL': config.MODEL.todict()}))

    # check that there are no overlapping files between train and val sets
    check_overlapping_files(config.POST.PATH.TRAIN, config.POST.PATH.VAL)

    # get features indices from feature extractor instance
    feature_indices = get_feature_indices(config.POST.PATH.TRAIN,
                                          config.POST.FEATURES)

    # get files indices from mixture info file
    file_indices = get_file_indices(config.POST.PATH.TRAIN)

    # seed for reproducibility
    torch.manual_seed(0)

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
        batch_size=config.MODEL.TRAIN.BATCHSIZE,
        shuffle=config.MODEL.TRAIN.SHUFFLE,
        num_workers=config.MODEL.TRAIN.NWORKERS,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.MODEL.TRAIN.BATCHSIZE,
        shuffle=config.MODEL.TRAIN.SHUFFLE,
        num_workers=config.MODEL.TRAIN.NWORKERS,
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
        hidden_config=config.MODEL.ARCHITECTURE,
        output_size=train_dataset.n_labels,
        dropout=config.MODEL.TRAIN.DROPOUT,
    )
    if config.MODEL.TRAIN.CUDA:
        model = model.cuda()

    # initialize criterion and optimizer
    criterion = getattr(torch.nn, config.MODEL.TRAIN.CRITERION)()
    optimizer = getattr(torch.optim, config.MODEL.TRAIN.OPTIMIZER)(
        params=model.parameters(),
        lr=config.MODEL.TRAIN.LEARNINGRATE,
        weight_decay=config.MODEL.TRAIN.WEIGHTDECAY,
    )

    # initialize early stopping object
    early_stopping = EarlyStopping(
        patience=config.MODEL.TRAIN.EARLYSTOP.PATIENCE,
        verbose=config.MODEL.TRAIN.EARLYSTOP.VERBOSE,
        delta=config.MODEL.TRAIN.EARLYSTOP.DELTA,
        checkpoint_dir=output_dir,
    )

    # main loop
    logging.info('Starting main loop...')
    train_losses = []
    val_losses = []
    start_time = time.time()
    for epoch in range(config.MODEL.TRAIN.EPOCHS):
        # train
        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            cuda=config.MODEL.TRAIN.CUDA,
        )

        # validate
        train_loss, val_loss = eval(
            model=model,
            criterion=criterion,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            load=config.POST.LOAD,
            cuda=config.MODEL.TRAIN.CUDA,
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
    plot_losses(train_losses, val_losses, output_dir)

    # close log file handler and rename model
    clear_logger()
    os.rename(output_dir, new_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('-i', '--input',
                        help=('Input YAML file.'))
    parser.add_argument('--all',
                        help=('Train all available models.'),
                        action='store_true')
    parser.add_argument('-f', '--force',
                        help=('Train already trained.'),
                        action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
    )

    if args.all:
        models_path = 'models'
        for root, folders, files in os.walk(models_path):
            if root == models_path:
                continue
            files = [file for file in files if file.endswith('.yaml')]
            if len(files) > 1:
                raise ValueError(f'More than one YAML file in {root}.')
            for file in files:
                main(os.path.join(root, file), args.force)

    else:
        main(args.input, args.force)
