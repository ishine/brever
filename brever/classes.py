import numpy as np
import scipy.signal
import random

from .utils import pca, frame
from .filters import mel_filterbank, gammatone_filterbank
from .mixture import make_mixture
from .io import load_random_target, load_brir, load_brirs
from . import features as features_module
from . import labels as labels_module


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


class BaseClass:
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
        output = (self.__class__.__module__ + '.' + self.__class__.__name__ +
                  ' instance:\n    ' +
                  '\n    '.join(': '.join((str(key), str(value)))
                                for key, value in attrs.items()))
        return output


class Filterbank(BaseClass):
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


class Framer(BaseClass):
    def __init__(self, frame_length, hop_length, window, center):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = window
        self.center = center

    def frame(self, x):
        return frame(x, frame_length=self.frame_length,
                     hop_length=self.hop_length, window=self.window,
                     center=self.center)


class FeatureExtractor(BaseClass):
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
            i_end = i_start + feature_set.shape[1]
            self.indices.append((i_start, i_end))
            i_start = i_end
        return np.hstack(output)


class LabelExtractor(BaseClass):
    def __init__(self, label):
        self.label = label

    def run(self, target, noise):
        label_func = getattr(labels_module, self.label)
        return label_func(target, noise, filtered=True, framed=True)


class RandomMixtureMaker(BaseClass):
    def __init__(self, rooms, angles_target, angles_directional, snrs,
                 snrs_directional_to_diffuse, colors_directional,
                 colors_diffuse, n_directional_sources, padding,
                 reflection_boundary, fs):
        self.rooms = rooms
        self.angles_target = angles_target
        self.angles_directional = angles_directional
        self.snrs = snrs
        self.snrs_directional_to_diffuse = snrs_directional_to_diffuse
        self.colors_directional = colors_directional
        self.colors_diffuse = colors_diffuse
        self.n_directional_sources = n_directional_sources
        self.padding = padding
        self.reflection_boundary = reflection_boundary
        self.fs = fs

    def make(self):
        angle = random.choice(self.angles_target)
        room = random.choice(self.rooms)
        snr = random.choice(self.snrs)
        color_diffuse = random.choice(self.colors_diffuse)
        n_directional_sources = random.choice(self.n_directional_sources)
        snrs_dir_to_diff = random.choices(self.snrs_directional_to_diffuse,
                                          k=n_directional_sources)
        colors_directional = random.choices(self.colors_directional,
                                            k=n_directional_sources)
        angles_directional = random.choices(self.angles_directional,
                                            k=n_directional_sources)
        target, filename = load_random_target()
        brir_target, fs = load_brir(room, angle)
        if fs != self.fs:
            raise ValueError(('the brir samplerate obtained from load_brir '
                              '(%s, %i) does not match the RandomMixtureMaker '
                              'instance samplerate attribute (%i vs %i)'
                              % (room, angle, fs, self.fs)))
        brirs_diffuse, fs = load_brirs(room)
        if fs != self.fs:
            raise ValueError(('the brir samplerate obtained from load_brirs '
                              '(%s) does not match the RandomMixtureMaker '
                              'instance samplerate attribute (%i vs %i)'
                              % (room, fs, self.fs)))
        brirs_directional, fs = load_brirs(room, angles_directional)
        if fs is not None and fs != self.fs:
            raise ValueError(('the brir samplerate obtained from load_brir '
                              '(%s, %s) does not match the RandomMixtureMaker '
                              'instance samplerate attribute (%i vs %i)'
                              % (room, angles_directional, fs, self.fs)))
        components = make_mixture(target=target,
                                  brir_target=brir_target,
                                  brirs_diffuse=brirs_diffuse,
                                  brirs_directional=brirs_directional,
                                  snr=snr,
                                  snrs_directional_to_diffuse=snrs_dir_to_diff,
                                  color_diffuse=color_diffuse,
                                  colors_directional=colors_directional,
                                  padding=self.padding,
                                  reflection_boundary=self.reflection_boundary,
                                  fs=self.fs)
        metadata = {
            'room': room,
            'filename': filename,
            'angle': angle,
            'snr': snr,
            'color_diffuse': color_diffuse,
            'n_directional_sources': n_directional_sources,
            'angles_directional': angles_directional,
            'snrs_dir_to_diff': snrs_dir_to_diff,
            'colors_directional': colors_directional,
        }
        return components, metadata
