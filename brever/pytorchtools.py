import numpy as np
import torch
import h5py
import os
import logging


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
                 decimation=1, feature_indices=None, file_indices=None):
        self.filepath = filepath
        self.datasets = None
        self.load = load
        self.transform = transform
        self.stack = stack
        self.decimation = decimation
        self.feature_indices = feature_indices
        self.file_indices = file_indices
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
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.n_samples


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.n_samples = len(x)
        self.n_features = x.shape[1]
        self.n_labels = y.shape[1]
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.x[index]), self.y[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_config, output_size, dropout,
                 momentum):
        super(Feedforward, self).__init__()
        self.operations = torch.nn.ModuleList()
        for item in hidden_config:
            if isinstance(item, int):
                self.operations.append(torch.nn.Linear(input_size, item))
                input_size = item
            elif item == 'ReLU':
                self.operations.append(torch.nn.ReLU())
            elif item == 'BN':
                self.operations.append(torch.nn.BatchNorm1d(input_size,
                                                            momentum=momentum))
            elif item == 'DO':
                self.operations.append(torch.nn.Dropout(dropout))
            else:
                raise ValueError(f'Unrecognized hidden operation, got {item}')
        self.operations.append(torch.nn.Linear(input_size, output_size))
        self.operations.append(torch.nn.Sigmoid())

    def forward(self, x):
        for operation in self.operations:
            x = operation(x)
        return x
