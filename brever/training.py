from itertools import permutations
from collections import deque
import logging
import time
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from .data import BreverBatchSampler, BreverDataLoader

eps = torch.finfo(torch.float32).eps


class TrainingTimer:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.steps_taken = None
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self.steps_taken = 0

    def step(self):
        self.steps_taken += 1

    def log(self):
        etl = self.estimated_time_left
        h, m, s = int(etl//3600), int((etl % 3600)//60), int(etl % 60)
        tps = int(self.time_per_step)
        log = f'Time /epoch: {tps}; ETA: {h} h {m} m {s} s'
        logging.info(log)

    def final_log(self):
        total_time = self.elapsed_time
        logging.info(f'Time spent: {int(total_time/3600)} h '
                     f'{int(total_time%3600/60)} m {int(total_time%60)} s')

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def time_per_step(self):
        return self.elapsed_time/self.steps_taken

    @property
    def steps_left(self):
        return self.total_steps - self.steps_taken

    @property
    def estimated_time_left(self):
        return self.time_per_step*self.steps_left


class LossLogger:
    def __init__(self, dirpath):
        self.train_loss = []
        self.val_loss = []
        self.dirpath = dirpath

    def add(self, train_loss, val_loss):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

    def log(self, epoch):
        train_loss = self.train_loss[-1]
        val_loss = self.val_loss[-1]
        logging.info(f'Epoch {epoch}: '
                     f'train loss: {train_loss:.4f}, '
                     f'val loss: {val_loss:.4f}')

    def plot(self):
        plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
        plt.rc('grid', color='w', linestyle='solid')
        fig, ax = plt.subplots()
        ax.plot(self.train_loss)
        ax.plot(self.val_loss)
        ax.legend(['training loss', 'validation loss'])
        ax.set_xlabel('epoch')
        ax.set_ylabel('error')
        ax.grid(True)
        plot_output_path = os.path.join(self.dirpath, 'training_curve.png')
        fig.tight_layout()
        fig.savefig(plot_output_path)
        plt.close(fig)

    def save(self):
        loss_path = os.path.join(self.dirpath, 'losses.npz')
        np.savez(loss_path, train=self.train_loss, val=self.val_loss)


class BreverTrainer:
    def __init__(self, model, train_dataset, val_dataset, dirpath,
                 batch_size=1, workers=0, epochs=100, learning_rate=1e-3,
                 weight_decay=0.0, cuda=True, criterion='MSELoss',
                 optimizer='Adam', early_stop=False, early_stop_patience=10,
                 convergence=False, convergence_window=10,
                 convergence_threshold=1.0e-6, grad_clip=0.0,
                 ignore_checkpoint=False):

        if early_stop and convergence:
            raise ValueError('cannot toggle both early_stop and convergence')

        self.model = model
        self.epochs = epochs
        self.cuda = cuda
        self.checkpoint_path = os.path.join(dirpath, 'checkpoint.pt')
        self.early_stop = early_stop
        self.convergence = convergence
        self.grad_clip = grad_clip
        self.ignore_checkpoint = ignore_checkpoint
        self.epochs_ran = 0

        # batch samplers
        self.train_batch_sampler = BreverBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
        )
        self.val_batch_sampler = BreverBatchSampler(
            dataset=val_dataset,
            batch_size=batch_size,
        )

        # dataloaders
        self.train_dataloader = BreverDataLoader(
            dataset=train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=workers,
        )
        self.val_dataloader = BreverDataLoader(
            dataset=val_dataset,
            batch_sampler=self.val_batch_sampler,
            num_workers=workers,
        )

        # criterion
        self.criterion = get_criterion(criterion)

        # optimizer
        self.optimizer = getattr(torch.optim, optimizer)(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # loss logger
        self.loss_logger = LossLogger(dirpath)

        # early stopper
        self.early_stopper = EarlyStopping(
            model=model,
            checkpoint_path=self.checkpoint_path,
            patience=early_stop_patience,
        )

        # convergence tracker
        self.convergence_tracker = ConvergenceTracker(
            window=convergence_window,
            threshold=convergence_threshold,
        )

        # timer
        self.timer = TrainingTimer(epochs)

    def run(self):
        # check for a checkpoint
        if not self.ignore_checkpoint and os.path.exists(self.checkpoint_path):
            logging.info('Checkpoint found')
            self.load_checkpoint()
            # if training was interrupted then resume training
            if self.epochs_ran < self.epochs:
                logging.info(f'Resuming training at epoch {self.epochs_ran}')
            else:
                logging.info('Model is already trained')
                return

        # initialize timer
        self.timer.start()
        for epoch in range(self.epochs_ran, self.epochs):
            # train
            train_loss = self.train()
            # evaluate
            val_loss = self.evaluate()
            # log losses
            self.loss_logger.add(train_loss, val_loss)
            self.loss_logger.log(epoch)
            # check stop criterion
            if self.stop_criterion(train_loss, val_loss):
                break
            # save checkpoint
            self.epochs_ran += 1
            self.save_checkpoint()
            # update time spent
            self.timer.step()
            self.timer.log()
        # plot and save losses
        self.timer.final_log()
        self.loss_logger.plot()
        self.loss_logger.save()

    def train(self):
        self.model.train()
        train_loss = 0
        for data, target in self.train_dataloader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            if self.grad_clip != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip,
                )
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(self.train_dataloader)
        return train_loss

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            for data, target in self.val_dataloader:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
            val_loss /= len(self.val_dataloader)
        return val_loss

    def stop_criterion(self, train_loss, val_loss):
        if self.early_stop:
            self.early_stopper(val_loss)
            if self.early_stopper.stop:
                logging.info('Early stopping now')
                logging.info(f'Best val loss: {self.early_stopper.min_loss}')
                return False
        else:
            self.save_checkpoint()
            if self.convergence:
                self.convergence_tracker(train_loss)
                if self.convergence_tracker.stop:
                    logging.info('Train loss has converged')
                    return False

    def save_checkpoint(self):
        state = {
            'epochs': self.epochs_ran,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': {
                'train': self.loss_logger.train_loss,
                'val': self.loss_logger.val_loss,
            }
        }
        torch.save(state, self.checkpoint_path)

    def load_checkpoint(self):
        state = torch.load(self.checkpoint_path)
        self.model.load_state_dict(state['model'])
        if self.cuda:
            self.model.cuda()
            # if the model was moved to cuda then the optimizer needs to be
            # reinitialized before loading the optimizer state dictionary
            # see https://github.com/pytorch/pytorch/issues/2830
            self.optimizer.__init__(
                params=self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        self.optimizer.load_state_dict(state['optimizer'])
        self.loss_logger.train_loss = state['losses']['train']
        self.loss_logger.val_loss = state['losses']['val']
        self.epochs_ran = state['epochs']


class EarlyStopping:
    def __init__(self, model, checkpoint_path, patience=10, verbose=True):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.min_loss = np.inf

    def __call__(self, loss):
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
            torch.save(self.model.state_dict(), self.checkpoint_path)


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

    def __call__(self, loss):
        self.losses.append(loss)
        if len(self.losses) == self.window:
            change = self.average_relative_change(self.losses)
            logging.info(f'Training curve relative change: {change:.2e}')
            if change < self.threshold:
                logging.info('Relative change has dropped below threshold')
                self.stop = True


class SISNR:
    def __call__(self, data, target):
        """
        Calculate SI-SNR with PIT.

        Parameters
        ----------
        data: tensor
            Estimated sources. Shape `(batch_size, sources, length)`
        target: tensor
            True sources. Shape `(batch_size, sources, length)`

        Returns
        -------
        si_snr : float
            SI-SNR.
        """
        # (B, S, L) = (batch_size, sources, length)
        data = data - data.mean(dim=2, keepdim=True)
        target = target - target.mean(dim=2, keepdim=True)

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
        loss = 0 - torch.mean(max_snr)
        return loss


class SNR:
    def __call__(self, data, target):
        """
        Calculate SNR without PIT.

        Parameters
        ----------
        data: tensor
            Estimated sources. Shape `(batch_size, sources, length)`.
        target: tensor
            True sources. Shape `(batch_size, sources, length)`

        Returns
        -------
        snr : float
            SNR.
        """
        # (B, S, L) = (batch_size, sources, length)
        snr = torch.sum(target**2, dim=-1) \
            / (torch.sum((target-data)**2, dim=-1) + eps)  # (B, S)
        snr = 10*torch.log10(snr + eps)  # (B, S)
        loss = -torch.mean(snr)
        return loss


def get_criterion(name):
    if name == 'SISNR':
        return SISNR()
    elif name == 'SNR':
        return SNR()
    else:
        return getattr(torch.nn, name)()
