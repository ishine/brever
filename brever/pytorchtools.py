import numpy as np
import torch
import h5py
import os
import logging


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


def evaluate(model, criterion, dataloader, load, cuda):
    model.eval()
    with torch.no_grad():
        if load:
            data, target = dataloader.dataset[:]
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
        if isinstance(mean, torch.Tensor):
            mean = mean.numpy()
        if isinstance(std, torch.Tensor):
            std = std.numpy()
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean)/self.std


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, load=False, transform=None, stack=0,
                 decimation=1, feature_indices=None, file_indices=None):
        self.filepath = filepath
        self.datasets = None
        self.load = load
        self.transform = transform
        self.stack = stack
        self.decimation = decimation
        self.feature_indices = feature_indices
        with h5py.File(self.filepath, 'r') as f:
            assert len(f['features']) == len(f['labels'])
            self.n_samples = len(f['features'])//decimation
            if feature_indices is None:
                self.n_features = f['features'].shape[1]
            else:
                self.n_features = sum(j-i for i, j in feature_indices)
            self.n_features *= (stack+1)
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
        # decimate
        if isinstance(index, int):
            index *= self.decimation
            if self.feature_indices is None:
                x = self.datasets[0][index]
            else:
                x = np.hstack([self.datasets[0][index, i:j]
                               for i, j in self.feature_indices])
            for k in range(self.stack):
                lag = k+1
                index_lagged = index-lag
                for i_file, _ in self.file_indices:
                    if i_file <= index and not i_file <= index-lag:
                        index_lagged = i_file
                        break
                if self.feature_indices is None:
                    lagged = self.datasets[0][index_lagged]
                else:
                    lagged = np.hstack([self.datasets[0][index_lagged, i:j]
                                        for i, j in self.feature_indices])
                x = np.hstack((x, lagged))
        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if start is not None:
                start *= self.decimation
            if stop is not None:
                stop *= self.decimation
            if step is None:
                step = self.decimation
            else:
                step *= self.decimation
            index = slice(start, stop, step)
            if self.feature_indices is None:
                x = self.datasets[0][index]
            else:
                x = np.hstack([self.datasets[0][index, i:j]
                               for i, j in self.feature_indices])
            for k in range(self.stack):
                lag = k+1
                if self.feature_indices is None:
                    lagged = self.datasets[0][:]
                else:
                    lagged = np.hstack([self.datasets[0][:, i:j]
                                        for i, j in self.feature_indices])
                lagged = np.roll(lagged, lag, axis=0)
                for i_file, _ in self.file_indices:
                    lagged[i_file:i_file+lag] = lagged[i_file+lag]
                lagged = lagged[index]
                x = np.hstack((x, lagged))
        else:
            raise ValueError((f'{type(self).__name__} does not support '
                              f'{type(index).__name__} indexing'))
        y = self.datasets[1][index]
        if self.transform:
            x = self.transform(x)
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
                 dropout_rate, batchnorm_toggle, batchnorm_momentum):
        super(Feedforward, self).__init__()
        self.operations = torch.nn.ModuleList()
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
