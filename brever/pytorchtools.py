import os
import logging
import pickle
import json

import yaml
import numpy as np
import torch
import h5py

from .utils import dct


def get_mean_and_std(dataset, dataloader, uniform_stats_features):
    if dataset.load:
        mean = dataset[:][0].mean(axis=0)
        std = dataset[:][0].std(axis=0)
    else:
        mean = 0
        for data, _ in dataloader:
            mean += data.mean(axis=0)
        mean /= len(dataloader)
        var = 0
        for data, _ in dataloader:
            var += ((data - mean)**2).mean(axis=0)
        var /= len(dataloader)
        std = var**0.5
        mean, std = mean.numpy(), std.numpy()
    for feature in uniform_stats_features:
        if feature not in dataset.features:
            raise ValueError(f'Uniform standardization feature "{feature}"" '
                             'is not in the list of selected features: '
                             f'{dataset.features}')
    mean, std = unify_stats(mean, std, dataset, uniform_stats_features)
    return mean, std


def get_files_mean_and_std(dataset, uniform_stats_features):
    means = []
    stds = []
    for i_start, i_end in dataset.file_indices:
        i_start = i_start // dataset.decimation
        i_end = i_end // dataset.decimation
        features, _ = dataset[i_start:i_end]
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        mean, std = unify_stats(mean, std, dataset, uniform_stats_features)
        means.append(mean)
        stds.append(std)
    return means, stds


def unify_stats(mean, std, dataset, uniform_stats_features):
    i_start = 0
    for feature, indices in zip(dataset.features, dataset.feature_indices):
        i_start_dataset, i_end_dataset = indices
        # i_start is the starting feature index in the stats vector
        # i_start_dataset is the starting feature index in the dataset
        i_end = i_start + i_end_dataset - i_start_dataset
        if feature in uniform_stats_features:
            feature_mean = mean[i_start:i_end].mean()
            feature_std = np.sqrt(
                np.mean(std[i_start:i_end]**2)
                + np.mean(mean[i_start:i_end]**2)
                - feature_mean**2
            )
            mean[i_start:i_end] = feature_mean
            std[i_start:i_end] = feature_std
        i_start = i_end
    return mean, std


def evaluate(model, criterion, dataloader, load, cuda):
    model.eval()
    with torch.no_grad():
        if load:
            data, target = dataloader.dataset[:]
            data, target = torch.from_numpy(data), torch.from_numpy(target)
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data.float(), target.float()
            output = model(data)
            loss = criterion(output, target)
            total_loss = loss.item()
        else:
            total_loss = 0
            for data, target in dataloader:
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = data.float(), target.float()
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
            total_loss /= len(dataloader)
    return total_loss


class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.min_loss = np.inf
        self.delta = delta

    def __call__(self, loss, model, checkpoint_path):
        score = -loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of '
                             f'{self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            if self.verbose:
                logging.info(f'Minimum validation loss decreased from '
                             f'{self.min_loss:.6f} to {loss:.6f}. '
                             f'Saving model.')
            self.min_loss = loss
            self.counter = 0
            torch.save(model.state_dict(), checkpoint_path)


class ProgressTracker:
    def __init__(self, strip=100, threshold=-5.5):
        self.strip = strip
        self.threshold = threshold
        self.losses = []
        self.stop = False

    def smooth(self, data, sigma=50):
        df = np.subtract.outer(np.arange(len(data)), np.arange(len(data)))
        filtering_mat = np.exp(-0.5*(df/sigma)**2)/(sigma*(2*np.pi)**0.5)
        filtering_mat = np.tril(filtering_mat)
        filtering_mat /= filtering_mat.sum(axis=1, keepdims=True)
        return filtering_mat@data

    def slope(self, data):
        x = np.arange(len(data))
        cov = lambda x, y: np.sum((x - x.mean())*(y - y.mean()))
        return cov(x, data)/cov(x, x)

    def __call__(self, loss, model, checkpoint_path):
        self.losses.append(loss)
        torch.save(model.state_dict(), checkpoint_path)
        if len(self.losses) >= self.strip:
            losses = self.smooth(self.losses)
            slope = self.slope(losses[-self.strip:])
            slope = np.log10(abs(slope))
            logging.info(f'Training curve slope (log10): {slope}')
            if slope < self.threshold:
                logging.info(f'Training curve slope has dropped below '
                             f'{self.threshold}.')
                self.stop = True


class TensorStandardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean)/self.std


class StateTensorStandardizer:
    def __init__(self, means, stds, state=0):
        self.means = means
        self.stds = stds
        self.state = 0

    def set_state(self, state):
        self.state = state

    def __call__(self, data):
        mean = self.means[self.state]
        std = self.stds[self.state]
        return (data - mean)/std


class H5Dataset(torch.utils.data.Dataset):
    '''
    Custom dataset class that loads HDF5 files created by create_dataset.py

    Performs decimation, context stacking and feature selection

    Inspired from an extensive discussion on the PyTorch forums about how to
    efficiently implement HDF5 dataset classes:
    https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16,
    '''
    def __init__(self, dirpath, features=None, labels=None, load=False,
                 transform=None, stack=0, decimation=1, dct_toggle=False,
                 n_dct=5, file_based_stats=False, prestack=False):
        if features is not None:
            features = sorted(features)
        if labels is not None:
            labels = sorted(labels)
        self.dirpath = dirpath
        self.features = features
        self.labels = labels
        self.load = load
        self.transform = transform
        self.stack = stack
        self.decimation = decimation
        self.dct_toggle = dct_toggle
        self.n_dct = n_dct
        self.file_based_stats = file_based_stats
        self.prestack = prestack
        self.filepath = os.path.join(dirpath, 'dataset.hdf5')
        self._prestacked = False
        self.datasets = None
        self.filenum_array = None
        logging.info('Getting file indices...')
        self.file_indices = self.get_file_indices()
        logging.info('Done.')
        with h5py.File(self.filepath, 'r') as f:
            assert len(f['features']) == len(f['labels'])
            # calculate number of samples
            self.n_samples = len(f['features'])//decimation
            # calculate number of features
            if self.features is None:
                self._n_current_features = f['features'].shape[1]
                self.feature_indices = [(0, self._n_current_features)]
            else:
                logging.info('Getting features indices...')
                self.feature_indices = self.get_feature_indices()
                logging.info('Done.')
                self._n_current_features = sum(j-i for i, j in
                                               self.feature_indices)
            if dct_toggle:
                stack = n_dct
            self.n_features = self._n_current_features*(stack + 1)
            # calculate number of labels
            if self.labels is None:
                self.n_labels = f['labels'].shape[1]
                self.label_indices = [(0, self.n_labels)]
            else:
                logging.info('Getting label indices...')
                self.label_indices = self.get_label_indices()
                logging.info('Done.')
                self.n_labels = sum(j-i for i, j in self.label_indices)
            if self.load:
                logging.info('Loading dataset into memory...')
                self.datasets = (f['features'][:], f['labels'][:])
                logging.info('Done.')
                self.filenum_array = f['indexes'][:]
                if self.prestack:
                    logging.info('Prestacking...')
                    self.datasets = self[:]
                    logging.info('Done.')
                    self._prestacked = True

    def get_feature_indices(self):
        pipes_path = os.path.join(self.dirpath, 'pipes.pkl')
        with open(pipes_path, 'rb') as f:
            featureExtractor = pickle.load(f)['featureExtractor']
        names = featureExtractor.features
        indices = featureExtractor.indices
        indices_dict = {name: lims for name, lims in zip(names, indices)}
        if 'itd_ic' in indices_dict.keys():
            start, end = indices_dict.pop('itd_ic')
            step = (end - start)//2
            indices_dict['itd'] = (start, start+step)
            indices_dict['ic'] = (start+step, end)
        feature_indices = [indices_dict[feature] for feature in self.features]
        return feature_indices

    def get_label_indices(self):
        pipes_path = os.path.join(self.dirpath, 'pipes.pkl')
        with open(pipes_path, 'rb') as f:
            labelExtractor = pickle.load(f)['labelExtractor']
        names = labelExtractor.labels
        indices = labelExtractor.indices
        indices_dict = {name: lims for name, lims in zip(names, indices)}
        label_indices = [indices_dict[label] for label in self.labels]
        return label_indices

    def get_file_indices(self):
        metadatas_path = os.path.join(self.dirpath, 'mixture_info.json')
        with open(metadatas_path, 'r') as f:
            metadatas = json.load(f)
            indices = [item['dataset_indices'] for item in metadatas]
        return indices

    def __getitem__(self, index):
        if self.datasets is None:
            f = h5py.File(self.filepath, 'r')
            if self.load:
                self.datasets = (f['features'][:], f['labels'][:])
                self.filenum_array = f['indexes'][:]
                if self.prestack:
                    self.datasets = self[:]
                    self._prestacked = True
            else:
                self.datasets = (f['features'], f['labels'])
                self.filenum_array = f['indexes']
        if isinstance(index, int):
            if self._prestacked:
                x, y = self.datasets[0][index], self.datasets[1][index]
                if self.transform:
                    if self.file_based_stats:
                        file_num = int(self.filenum_array[index])
                        self.transform.set_state(file_num)
                    x = self.transform(x)
            else:
                # features
                x = np.empty(self.n_features)
                # decimate
                index *= self.decimation
                # frame at current time index
                count = 0
                for i, j in self.feature_indices:
                    i_ = count
                    j_ = count+j-i
                    x[i_:j_] = self.datasets[0][index, i:j]
                    count = j_
                # frames at previous time indexes
                if self.stack > 0 or (self.transform
                                      and self.file_based_stats):
                    file_num = int(self.filenum_array[index])
                if self.stack > 0:
                    x_context = np.zeros((self.stack,
                                          self._n_current_features))
                    # first find the starting index of the current file
                    i_file_start, _ = self.file_indices[file_num]
                    # then add context stacking
                    for k in range(self.stack):
                        # if context overlaps previous file then replicate
                        i_lag = max(index-k-1, i_file_start)
                        count_context_k = 0
                        for i, j in self.feature_indices:
                            i_ = count_context_k
                            j_ = count_context_k+j-i
                            x_context[k, i_:j_] = self.datasets[0][i_lag, i:j]
                            count_context_k = j_
                    # perform dct
                    if self.dct_toggle:
                        x_context = dct(x_context, self.n_dct)
                    x[count:] = x_context.flatten()
                if self.transform:
                    if self.file_based_stats:
                        self.transform.set_state(file_num)
                    x = self.transform(x)
                # labels
                y = np.empty(self.n_labels)
                count = 0
                for i, j in self.label_indices:
                    i_ = count
                    j_ = count+j-i
                    y[i_:j_] = self.datasets[1][index, i:j]
                    count = j_
        elif isinstance(index, slice):
            indexes = list(range(self.n_samples))[index]
            x = np.empty((len(indexes), self.n_features))
            y = np.empty((len(indexes), self.n_labels))
            for i, j in enumerate(indexes):
                x[i, :], y[i, :] = self.__getitem__(j)
        else:
            raise ValueError(f'{type(self).__name__} does not support '
                             f'{type(index).__name__} indexing')
        return x, y

    def __len__(self):
        return self.n_samples


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, output_size, n_layers, dropout_toggle,
                 dropout_rate, dropout_input, batchnorm_toggle,
                 batchnorm_momentum, scale_capacity):
        super(Feedforward, self).__init__()
        self.operations = torch.nn.ModuleList()
        if dropout_input:
            self.operations.append(torch.nn.Dropout(dropout_rate))
        if dropout_toggle and scale_capacity:
            hidden_size = int(round(input_size/(1-dropout_rate)))
        else:
            hidden_size = input_size
        for i in range(n_layers):
            if i == 0:
                start_size = input_size
            else:
                start_size = hidden_size
            self.operations.append(torch.nn.Linear(start_size, hidden_size))
            if batchnorm_toggle:
                self.operations.append(
                    torch.nn.BatchNorm1d(hidden_size,
                                         momentum=batchnorm_momentum))
            self.operations.append(torch.nn.ReLU())
            if dropout_toggle:
                self.operations.append(torch.nn.Dropout(dropout_rate))
        self.operations.append(torch.nn.Linear(hidden_size, output_size))
        self.operations.append(torch.nn.Sigmoid())

    @classmethod
    def build(cls, args_path):
        with open(args_path, 'r') as f:
            arguments = yaml.safe_load(f)
        model = cls(**arguments)
        return model

    def forward(self, x):
        for operation in self.operations:
            x = operation(x)
        return x
