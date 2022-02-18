import math

import numpy as np
import scipy.fft
import torch
from torchaudio.functional import lfilter

from .utils import pad

eps = np.finfo(float).eps


class FeatureExtractor:

    def __init__(self, features, hop_length=256, fs=16e3):
        self.features = features
        self.hop_length = hop_length
        self.fs = fs
        self.indices = None

    def __call__(self, x):
        output = []
        self.indices = {}
        i_start = 0
        for feature in self.features:
            data = self.calc_feature(x, feature)
            output.append(data)
            i_end = i_start + len(data)
            self.indices[feature] = (i_start, i_end)
            i_start = i_end
        return torch.cat(output)

    def calc_feature(self, x, feature):
        if feature == 'ild':
            return self.ild(x)
        elif feature == 'ipd':
            return self.ipd(x)
        elif feature == 'ic':
            return self.ic(x)
        elif feature == 'fbe':
            return self.fbe(x)
        elif feature == 'logfbe':
            return self.fbe(x, compression='log')
        elif feature == 'cubicfbe':
            return self.fbe(x, compression='cubic')
        elif feature == 'pdf':
            return self.fbe(x, normalize=True)
        elif feature == 'logpdf':
            return self.fbe(x, normalize=True, compression='log')
        elif feature == 'cubicpdf':
            return self.fbe(x, normalize=True, compression='cubic')
        elif feature == 'mfcc':
            return self.fbe(x, compression='log', dct=True)
        elif feature == 'cubicmfcc':
            return self.fbe(x, compression='cubic', dct=True)
        elif feature == 'pdfcc':
            return self.fbe(x, normalize=True, compression='log', dct=True)
        else:
            raise ValueError(f'unrecognized feature, got {feature}')

    def fbe(self, x, normalize=False, compression='none', dct=False, n_dct=14,
            dct_type=2, dct_norm='ortho', return_dc=False, return_deltas=True,
            return_double_deltas=True):
        """
        Filterbank energies.

        Calculates the energy in each time-frequency unit. Supports a series of
        compression and normalization options to obtain MFCC or PDF features.

        Parameters
        ----------
        x : array_like
            Input signal. Shape `(channels, bins, frames)`.
        normalize : bool, optional
            Whether to normalize along frequency axis. Default is `False`.
        compression : {'log', 'cubic', 'none'}, optional
            Compression type. Default is `'none'`.
        dct : bool, optional
            Wheter to apply DCT compression along the frequency axis. Default
            is `False`.
        n_dct : int, optional
            Number of DCT coefficients to return, including DC term. Ignored if
            `dct` is `False`. Default is 14.
        dct_type : {1, 2, 3, 4}, optional
            Type of DCT. Ignored if `dct` is `False`. Default is 2.
        dct_norm : {'backward', 'ortho', 'forward'}, optional
            Normalization mode for the DCT. Ignored if `dct` is `False`.
            Default is `'ortho'`.
        return_dc : bool, optional
            Whether to return the DC term. Ignored if `dct` is `False`.
            Default is `False`.
        return_deltas : bool or None, optional
            Whether to return first order difference along the frame axis.
            Ignored if `dct` is `False`. Default is `True`.
        return_double_deltas : bool or None, optional
            Whether to return second order difference along the frame axis.
            Ignored if `dct` is `False`. Default is `True`.

        Returns
        -------
        fbe : array_like
            Filterbank energies.
        """
        mag, phase = x
        assert mag.shape[0] == phase.shape[0] == 2
        # calculate energy
        out = mag.pow(2).mean(0)  # (bins, frames).
        # normalize
        if normalize:
            out /= out.sum(0, keepdims=True) + eps
        # compress
        if compression == 'log':
            out = torch.log(out + eps)
        elif compression == 'cubic':
            out = out.pow(1/3)
        elif compression != 'none':
            raise ValueError('compression must be log, cubic or none, got '
                             f'{compression}')
        # apply dct
        if dct:
            out = out.numpy()
            out = scipy.fft.dct(out, axis=0, type=dct_type, norm=dct_norm)
            if return_dc:
                out = np.take(out, range(n_dct), axis=0)
            else:
                out = np.take(out, range(1, n_dct), axis=0)
            present = out
            if return_deltas:
                diff = np.diff(present, n=1, axis=1)
                diff = pad(diff, n=1, axis=1, where='left')
                out = np.concatenate((out, diff), axis=0)
            if return_double_deltas:
                diff = np.diff(present, n=2, axis=1)
                diff = pad(diff, n=2, axis=1, where='left')
                out = np.concatenate((out, diff), axis=0)
            out = torch.from_numpy(out)
        return out

    def ild(self, x):
        """
        Interaural level difference.

        Calculates the interaural level difference between two channels.

        Parameters
        ----------
        x : array_like
            Input signal. Shape `(channels, bins, frames)`.

        Returns
        -------
        ild : array_like
            Interaural level difference.
        """
        mag, phase = x
        assert mag.shape[0] == phase.shape[0] == 2
        return 20*torch.log10(mag[1]/mag[0])

    def ipd(self, x):
        """
        Interaural phase difference.

        Calculates the interaural phase difference between two channels.

        Parameters
        ----------
        x : array_like
            Input signal. Shape `(channels, bins, frames)`.

        Returns
        -------
        ipd : array_like
            Interaural phase difference.
        """
        mag, phase = x
        assert mag.shape[0] == phase.shape[0] == 2
        return phase[1] - phase[0]

    def ic(self, x, tau=10e-3):
        """
        Interaural coherence.

        Calculates the interaural coherence between two channels.

        Parameters
        ----------
        x : array_like
            Input signal. Shape `(channels, bins, frames)`.
        tau : float
            Time constant for the exponentially-weighted auto- and cross-power
            spectra in seconds.

        Returns
        -------
        ic : array_like
            Interaural coherence.
        """
        mag, phase = x
        assert mag.shape[0] == phase.shape[0] == 2
        alpha = math.exp(-self.hop_length/(tau*self.fs))
        x_ll = mag[0].pow(2)
        x_rr = mag[1].pow(2)
        x_lr = mag[0]*mag[1]*torch.exp(1j*(phase[0]-phase[1]))
        phi_ll, phi_rr, phi_lr_real, phi_lr_imag = lfilter(
            torch.stack([x_ll, x_rr, x_lr.real, x_lr.imag]),
            a_coeffs=torch.tensor([1, alpha]),
            b_coeffs=torch.tensor([1-alpha, 0]),
        )
