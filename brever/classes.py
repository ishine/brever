import numpy as np
import scipy.signal
import random

from .utils import pca, frame
from .filters import mel_filterbank, gammatone_filterbank
from . import features as features_module
from .labels import irm
from .mixture import make as make_mixture
from .io import load_random_target, load_brir, load_brirs


class Standardizer:
    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, X):
        self.means = X.mean(axis=self.axis)
        self.stds = X.std(axis=self.axis)

    def transform(self, X):
        means = np.expand_dims(self.means, axis=self.axis)
        stds = np.expand_dims(self.stds, axis=self.axis)
        return (X - means)/stds


class PCA:
    def __init__(self, n_components=None, pve=None):
        self.n_components = n_components
        self.pve = pve

    def fit(self, X):
        components, ve, means = pca(X, n_components=self.n_components,
                                    pve=self.pve)
        self.components = components
        self.variance_explained = ve
        self.means = means

    def transform(self, X):
        return (X - self.means) @ self.components


class PipeBaseClass:
    def __init__(self):
        pass

    def __str__(self):
        attrs = self.__dict__
        for key, value in attrs.items():
            if isinstance(value, (list, dict, tuple)) and len(value) > 10:
                attrs[key] = '%s with length %i' % (type(value).__name__,
                                                    len(value))
            elif isinstance(value, np.ndarray):
                attrs[key] = 'numpy array with shape %s' % str(value.shape)
        output = (self.__class__.__name__ + ':\n    ' +
                  '\n    '.join(': '.join((str(key), str(value)))
                                for key, value in attrs.items()))
        return output


class Filterbank(PipeBaseClass):
    def __init__(self, kind, n_filters, f_min, f_max, fs, order):
        self.kind = kind
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.fs = fs
        self.order = order
        if kind == 'mel':
            b, a, fc = mel_filterbank(n_filters, f_min, f_max, fs, order)
        elif kind == 'gammatone':
            b, a, fc = gammatone_filterbank(n_filters, f_min, f_max, fs, order)
        else:
            raise ValueError('filter_type must be mel or gammatone')
        self.b = b
        self.a = a
        self.fc = fc

    def filt(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_samples, n_channels = x.shape
        x_filt = np.zeros((n_samples, self.n_filters, n_channels))
        for i in range(self.n_filters):
            x_filt[:, i, :] = scipy.signal.lfilter(self.b[i], self.a[i], x,
                                                   axis=0)
        return x_filt.squeeze()


class Framer(PipeBaseClass):
    def __init__(self, frame_length, hop_length, window, center):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = window
        self.center = center

    def frame(self, x):
        return frame(x, frame_length=self.frame_length,
                     hop_length=self.hop_length, window=self.window,
                     center=self.center)


class FeatureExtractor(PipeBaseClass):
    def __init__(self, features):
        self.features = features
        self.indices = None

    def run(self, x):
        output = []
        for feature in self.features:
            feature_func = getattr(features_module, feature)
            output.append(feature_func(x, filtered=True,  framed=True))
        self.indices = []
        i_start = 0
        for feature_set in output:
            i_end = i_start + len(feature_set)
            self.indices.append((i_start, i_end))
            i_start = i_end
        return np.hstack(output)


class LabelExtractor(PipeBaseClass):
    def __init__(self):
        pass

    def run(self, target, noise):
        return irm(target, noise, filtered=True, framed=True)


class RandomMixtureMaker(PipeBaseClass):
    def __init__(self, rooms, angles, snrs, padding,
                 reflection_boundary, max_itd, fs):
        self.rooms = rooms
        self.angles = angles
        self.snrs = snrs
        self.padding = padding
        self.reflection_boundary = reflection_boundary
        self.max_itd = max_itd
        self.fs = fs

    def make(self):
        angle = random.choice(self.angles)
        room = random.choice(self.rooms)
        snr = random.choice(self.snrs)
        target, filename = load_random_target()
        brir, fs = load_brir(room, angle)
        if fs != self.fs:
            raise ValueError(('the brir samplerate does not match the ',
                              'RandomMixtureMaker instance samplerate '
                              'attribute'))
        brirs, fs = load_brirs(room)
        if fs != self.fs:
            raise ValueError(('the brirs samplerate does not match the ',
                              'RandomMixtureMaker instance samplerate '
                              'attribute'))
        components = make_mixture(target, brir, brirs, snr,
                                  padding=self.padding,
                                  reflection_boundary=self.reflection_boundary,
                                  max_itd=self.max_itd, fs=self.fs)
        mix, target_reverb, target_early, target_late, noise = components
        return Mixture(components, angle, room, snr, filename)


class Mixture(PipeBaseClass):
    def __init__(self, components, angle, room, snr, filename):
        mix, target_reverb, target_early, target_late, noise = components
        self.mix = mix
        self.foreground = target_early
        self.background = target_late + noise
        self.angle = angle
        self.room = room
        self.snr = snr
        self.filename = filename
