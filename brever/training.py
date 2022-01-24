import logging

import numpy as np
import torch


def evaluate(model, criterion, dataloader, cuda):
    model.eval()
    with torch.no_grad():
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
        x = x - x.mean()
        y = data - data.mean()
        return np.sum(x*y)/np.sum(x*x)

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
