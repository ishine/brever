import numpy as np
import torch
import h5py


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print((f'EarlyStopping counter: {self.counter} out of '
                   '{self.patience}'))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print((f'Validation loss decreased ({self.val_loss_min:.6f} --> '
                   '{val_loss:.6f}).  Saving model ...'))
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, load=False, transform=None):
        self.filepath = filepath
        self.datasets = None
        self.load = load
        self.transform = transform
        with h5py.File(self.filepath, 'r') as f:
            assert len(f['features']) == len(f['labels'])
            self.n_samples = len(f['features'])
            self.n_features = f['features'].shape[1]
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
        x, y = self.datasets[0][index], self.datasets[1][index]
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.n_samples
