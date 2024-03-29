import math

import numpy as np
import scipy.fft
import torch
from torchaudio.functional import lfilter

from .utils import pad


eps = torch.finfo().eps


class FeatureExtractor:

    def __init__(self, features, mel_fb, hop_length=256, fs=16e3):
        self.features = sorted(features)
        self.mel_fb = mel_fb
        self.hop_length = hop_length
        self.fs = fs
        self.indices = None
        # _feature_dict is a map containing which function to call and how many
        #  features there are for each feature name
        self._feature_dict = {
            'ild': {
                'func': self.ild,
                'n': self.mel_fb.n_filters,
            },
            'ipd': {
                'func': self.ipd,
                'n': self.mel_fb.n_filters,
            },
            'ic': {
                'func': self.ic,
                'n': self.mel_fb.n_filters,
            },
            'fbe': {
                'func': self.fbe,
                'n': self.mel_fb.n_filters,
            },
            'logfbe': {
                'func': lambda x: self.fbe(
                    x,
                    compression='log',
                ),
                'n': self.mel_fb.n_filters,
            },
            'cubicfbe': {
                'func': lambda x: self.fbe(
                    x,
                    compression='cubic',
                ),
                'n': self.mel_fb.n_filters,
            },
            'pdf': {
                'func': lambda x: self.fbe(
                    x,
                    normalize=True,
                ),
                'n': self.mel_fb.n_filters,
            },
            'logpdf': {
                'func': lambda x: self.fbe(
                    x,
                    normalize=True,
                    compression='log',
                ),
                'n': self.mel_fb.n_filters,
            },
            'cubicpdf': {
                'func': lambda x: self.fbe(
                    x,
                    normalize=True,
                    compression='cubic',
                ),
                'n': self.mel_fb.n_filters,
            },
            'mfcc': {
                'func': lambda x: self.fbe(
                    x,
                    compression='log',
                    dct=True,
                ),
                'n': 13,
            },
            'cubicmfcc': {
                'func': lambda x: self.fbe(
                    x,
                    compression='cubic',
                    dct=True,
                ),
                'n': 13,
            },
            'pdfcc': {
                'func': lambda x: self.fbe(
                    x,
                    normalize=True,
                    compression='log',
                    dct=True,
                ),
                'n': 13,
            },
        }

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

    def _get_feature_info(self, feature):
        try:
            out = self._feature_dict[feature]
        except KeyError:
            raise ValueError(f'unrecognized feature, got {feature}')
        return out

    @property
    def n_features(self):
        out = 0
        for feature in self.features:
            out += self._get_feature_info(feature)['n']
        return out

    def calc_feature(self, x, feature):
        mag, phase = x
        assert mag.shape[0] == phase.shape[0] == 2
        feature_func = self._get_feature_info(feature)['func']
        return feature_func(x)

    def fbe(self, x, normalize=False, compression='none', dct=False, n_dct=14,
            dct_type=2, dct_norm='ortho', return_dc=False, return_deltas=True,
            return_double_deltas=True):
        """
        Filterbank energies.

        Calculates the energy in each time-frequency unit. Supports a series of
        compression and normalization options to obtain MFCC or PDF features.

        Parameters
        ----------
        x : tuple of array_like
            Magnitude and phase. Each of shape `(channels, bins, frames)`.
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
        # calculate energy
        out = mag.pow(2).mean(0)  # (bins, frames)
        # gather by filters
        out = self.mel_fb(out)
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
        x : tuple of array_like
            Magnitude and phase. Each of shape `(channels, bins, frames)`.

        Returns
        -------
        ild : array_like
            Interaural level difference.
        """
        mag, phase = x
        ild = 20*torch.log10((mag[1]+eps)/(mag[0]+eps))
        return self.mel_fb(ild)

    def ipd(self, x):
        """
        Interaural phase difference.

        Calculates the interaural phase difference between two channels.

        Parameters
        ----------
        x : tuple of array_like
            Magnitude and phase. Each of shape `(channels, bins, frames)`.

        Returns
        -------
        ipd : array_like
            Interaural phase difference.
        """
        mag, phase = x
        ipd = phase[1] - phase[0]
        return self.mel_fb(ipd)

    def ic(self, x, tau=10e-3):
        """
        Interaural coherence.

        Calculates the interaural coherence between two channels.

        Parameters
        ----------
        x : tuple of array_like
            Magnitude and phase. Each of shape `(channels, bins, frames)`.
        tau : float
            Time constant for the exponentially-weighted auto- and cross-power
            spectra in seconds.

        Returns
        -------
        ic : array_like
            Interaural coherence.
        """
        mag, phase = x
        alpha = math.exp(-self.hop_length/(tau*self.fs))
        x_ll = mag[0].pow(2)
        x_rr = mag[1].pow(2)
        x_lr = mag[0]*mag[1]*torch.exp(1j*(phase[0]-phase[1]))
        phi_ll, phi_rr, phi_lr_real, phi_lr_imag = lfilter(
            torch.stack([x_ll, x_rr, x_lr.real, x_lr.imag]),
            a_coeffs=torch.tensor([1, -alpha]),
            b_coeffs=torch.tensor([1-alpha, 0]),
        )
        phi_lr_mag = (phi_lr_real.pow(2) + phi_lr_imag.pow(2)).sqrt()
        ic = phi_lr_mag.pow(2)/(phi_ll*phi_rr)
        return self.mel_fb(ic).sqrt()
