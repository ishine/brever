from itertools import permutations
from collections import deque
import logging

import numpy as np
import torch

eps = torch.finfo(torch.float32).eps


def evaluate(model, criterion, dataloader, cuda):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for data, target in dataloader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
        total_loss /= len(dataloader)
    return total_loss


class EarlyStopping:
    def __init__(self, patience=20, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.min_loss = np.inf

    def __call__(self, loss, model, checkpoint_path):
        score = -loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
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


class ConvergenceTracker:
    def __init__(self, window=10, threshold=1e-6):
        self.window = window
        self.threshold = threshold
        self.losses = deque(maxlen=window)
        self.stop = False

    def average_relative_change(self, data):
        x = np.arange(len(data))
        x_center = x - np.mean(x)
        y_center = data - np.mean(data)
        slope = np.sum(x_center*y_center)/np.sum(x_center*x_center)
        return slope/np.mean(data)

    def __call__(self, loss, model, checkpoint_path):
        self.losses.append(loss)
        if len(self.losses) == self.window:
            change = self.average_relative_change(self.losses)
            logging.info(f'Training curve relative change: {change:.2e}')
            if change < self.threshold:
                logging.info('Relative change has dropped below threshold. '
                             'Interrupting training')
                self.stop = True


class SISNR:
    def __call__(self, data, target):
        """
        Calculate SI-SNR with PIT.

        Parameters
        ----------
        data: tensor
            Estimated sources. Shape `(batch_size, sources, lenght)`
        target: tensor
            True sources. Shape `(batch_size, sources, lenght)`

        Returns
        -------
        si_snr : float
            SI-SNR.
        """
        # (B, S, L) = (batch_size, sources, lenght)
        s_hat = torch.unsqueeze(data, dim=1)  # (B, 1, S, L)
        s = torch.unsqueeze(target, dim=2)  # (B, S, 1, L)
        s_target = torch.sum(s_hat*s, dim=3, keepdim=True)*s \
            / torch.sum(s**2, dim=3, keepdim=True)  # (B, S, S, L)
        e_noise = s_hat - s_target  # (B, S, S, L)
        si_snr = torch.sum(s_target**2, dim=3) \
            / (torch.sum(e_noise ** 2, dim=3) + eps)  # (B, S, S, L)
        si_snr = 10*torch.log10(si_snr + eps)

        S = data.shape[1]
        perms = data.new_tensor(list(permutations(range(S))), dtype=torch.long)
        index = torch.unsqueeze(perms, 2)
        one_hot = data.new_zeros((*perms.size(), S)).scatter_(2, index, 1)
        snr_set = torch.einsum('bij,pij->bp', [si_snr, one_hot])
        max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
        max_snr /= S
        return max_snr


def get_criterion(name):
    if name == 'SISNR':
        return SISNR()
    else:
        return getattr(torch.nn, name)()
