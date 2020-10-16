import numpy as np
import torch
import h5py
import os
import logging

from .utils import dct_compress


def get_mean_and_std(dataloader, load):
    if load:
        mean = dataloader.dataset[:][0].mean(axis=0)
        std = dataloader.dataset[:][0].std(axis=0)
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
    def __init__(self, patience=7, verbose=True, delta=0,
                 checkpoint_dir=''):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info((f'EarlyStopping counter: {self.counter} out of '
                              f'{self.patience}'))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logging.info((f'Validation loss decreased from '
                          f'{self.val_loss_min:.6f} to {val_loss:.6f}. '
                          f'Saving model...'))
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss


class TensorStandardizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean)/self.std


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, load=False, transform=None, stack=0,
                 decimation=1, feature_indices=None, file_indices=None,
                 dct=False, n_dct=5):
        if n_dct < 1:
            raise ValueError('n_dct must be higher than 1')
        self.filepath = filepath
        self.datasets = None
        self.load = load
        self.transform = transform
        self.stack = stack
        self.decimation = decimation
        self.dct = dct
        self.n_dct = n_dct
        with h5py.File(self.filepath, 'r') as f:
            assert len(f['features']) == len(f['labels'])
            self.n_samples = len(f['features'])//decimation
            if feature_indices is None:
                self._n_current_features = f['features'].shape[1]
                self.feature_indices = [(0, self._n_current_features)]
            else:
                self._n_current_features = sum(j-i for i, j in feature_indices)
                self.feature_indices = feature_indices
            if dct:
                stack = min(n_dct-1, stack)
            self.n_features = self._n_current_features*(stack + 1)
            self.n_labels = f['labels'].shape[1]
            if self.load:
                self.datasets = (f['features'][:], f['labels'][:])
        if file_indices is None:
            file_indices = [(0, self.n_samples)]
        self.file_indices = file_indices

    def __getitem__(self, index):
        if self.datasets is None:
            f = h5py.File(self.filepath, 'r')
            if self.load:
                self.datasets = (f['features'][:], f['labels'][:])
            else:
                self.datasets = (f['features'], f['labels'])
        if isinstance(index, int):
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
            if self.stack > 0:
                x_context = np.empty((self.stack, self._n_current_features))
                # first check if a file starts during delay window
                index_min = index-self.stack
                for i_file, _ in self.file_indices:
                    if index_min < i_file <= index:
                        index_min = i_file
                        break
                    # stop searching if current time index is reached
                    elif i_file > index:
                        break
                # then add context stacking
                for k in range(self.stack):
                    # if context overlaps previous file then replicate
                    index_lag = max(index-k-1, index_min)
                    count_context_k = 0
                    for i, j in self.feature_indices:
                        i_ = count_context_k
                        j_ = count_context_k+j-i
                        x_context[k, i_:j_] = self.datasets[0][index_lag, i:j]
                        count_context_k = j_
                # perform dct
                if self.dct and self.n_dct-1 < self.stack:
                    x_context = dct_compress(x_context, self.n_dct)
                x[count:] = x_context.flatten()
            if self.transform:
                x = self.transform(x)
            y = self.datasets[1][index]
        elif isinstance(index, slice):
            indexes = list(range(self.n_samples))[index]
            x = np.empty((len(indexes), self.n_features))
            y = np.empty((len(indexes), self.n_labels))
            for i, j in enumerate(indexes):
                x[i, :], y[i, :] = self.__getitem__(j)
        else:
            raise ValueError((f'{type(self).__name__} does not support '
                              f'{type(index).__name__} indexing'))
        return x, y

    def __len__(self):
        return self.n_samples


class DummyH5Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, load):
        self.filepath = filepath
        self.load = load
        self.file = None
        with h5py.File(filepath, 'r') as f:
            self.n_samples = len(f['features'])
            self.n_features = f['features'].shape[1]
            self.n_labels = f['labels'].shape[1]

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.filepath, 'r')
        return (
            self.file['features'][index],
            self.file['labels'][index],
        )

    def __len__(self):
        return self.n_samples


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, output_size, n_layers, dropout_toggle,
                 dropout_rate, dropout_input, batchnorm_toggle,
                 batchnorm_momentum):
        super(Feedforward, self).__init__()
        self.operations = torch.nn.ModuleList()
        if dropout_input:
            self.operations.append(torch.nn.Dropout(dropout_rate))
        for i in range(n_layers):
            self.operations.append(torch.nn.Linear(input_size, input_size))
            if batchnorm_toggle:
                self.operations.append(
                    torch.nn.BatchNorm1d(input_size,
                                         momentum=batchnorm_momentum))
            self.operations.append(torch.nn.ReLU())
            if dropout_toggle:
                self.operations.append(torch.nn.Dropout(dropout_rate))
        self.operations.append(torch.nn.Linear(input_size, output_size))
        self.operations.append(torch.nn.Sigmoid())

    def forward(self, x):
        for operation in self.operations:
            x = operation(x)
        return x
