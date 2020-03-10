import numpy as np
import scipy.signal
import random
import re

from .utils import pca, frame
from .filters import mel_filterbank, gammatone_filterbank
from .mixture import make_mixture, colored_noise
from .io import load_random_target, load_brir, load_brirs, load_random_noise
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
        output = (self.__class__.__module__ + '.' + self.__class__.__name__
                  + ' instance:\n    '
                  + '\n    '.join(': '.join((str(key), str(value)))
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
            output.append(feature_func(x, filtered=True, framed=True))
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
                 snrs_directional_to_diffuse, types_directional,
                 types_diffuse, n_directional_sources, padding,
                 reflection_boundary, fs, lims):
        self.rooms = rooms
        self.angles_target = angles_target
        self.angles_directional = angles_directional
        self.snrs = snrs
        self.snrs_directional_to_diffuse = snrs_directional_to_diffuse
        self.types_directional = types_directional
        self.types_diffuse = types_diffuse
        self.n_directional_sources = n_directional_sources
        self.padding = padding
        self.reflection_boundary = reflection_boundary
        self.fs = fs
        self.lims = lims

    def make(self):
        angle = random.choice(self.angles_target)
        room = random.choice(self.rooms)
        snr = random.choice(self.snrs)
        type_diff = random.choice(self.types_diffuse)
        n_dir_sources = random.choice(self.n_directional_sources)
        snrs_dir_to_diff = random.choices(self.snrs_directional_to_diffuse,
                                          k=n_dir_sources)
        types_dir = random.choices(self.types_directional,
                                   k=n_dir_sources)
        angles_dir = random.choices(self.angles_directional,
                                    k=n_dir_sources)
        x_target, file_target = load_random_target()
        n_samples = len(x_target) + 2*round(self.padding*self.fs)
        x_diff, file_diff, i_diff = self._load_noises(type_diff, n_samples)
        xs_dir, files_dir, is_dir = self._load_noises(types_dir, n_samples)
        brir_target = self._load_brirs(room, angle)
        brirs_diff = self._load_brirs(room)
        brirs_dir = self._load_brirs(room, angles_dir)
        components = make_mixture(x_target=x_target,
                                  brir_target=brir_target,
                                  brirs_diffuse=brirs_diff,
                                  brirs_directional=brirs_dir,
                                  snr=snr,
                                  snrs_directional_to_diffuse=snrs_dir_to_diff,
                                  x_diffuse=x_diff,
                                  xs_directional=xs_dir,
                                  padding=self.padding,
                                  reflection_boundary=self.reflection_boundary,
                                  fs=self.fs)
        metadata = {
            'room': room,
            'target_filename': file_target,
            'target_angle': angle,
            'snr': snr,
            'n_directional_sources': n_dir_sources,
            'directional_noises_filenames': files_dir,
            'directional_sources_angles': angles_dir,
            'snrs_dir_to_diff': snrs_dir_to_diff,
            'diffuse_noise_filename': file_diff,
            'directional_sources_indices': is_dir,
            'difuse_noise_indices': i_diff,
        }
        return components, metadata

    def _load_brirs(self, room, angles=None):
        if angles is None or isinstance(angles, list):
            brirs, fs = load_brirs(room, angles)
            if fs is not None and fs != self.fs:
                raise ValueError(('the brir samplerate obtained from '
                                  'load_brirs(%s, %s) does not match the '
                                  'RandomMixtureMaker instance samplerate '
                                  'attribute (%i vs %i)'
                                  % (room, angles, fs, self.fs)))
            return brirs
        else:
            brir, fs = load_brir(room, angles)
            if fs is not None and fs != self.fs:
                raise ValueError(('the brir samplerate obtained from '
                                  'load_brir(%s, %s) does not match the '
                                  'RandomMixtureMaker instance samplerate '
                                  'attribute (%i vs %i)'
                                  % (room, angles, fs, self.fs)))
            return brir

    def _load_noises(self, types, n_samples):
        if isinstance(types, list):
            zipped = [self._load_noises(type_, n_samples) for type_ in types]
            xs, filepaths, indicess = zip(*zipped)
            return xs, filepaths, indicess
        else:
            type_ = types
            if type_.startswith('noise_'):
                color = re.match('^noise_(.*)$', type_).group(1)
                x = colored_noise(color, n_samples)
                filepath = None
                indices = None
            elif type_.startswith('dcase_'):
                x, filepath, indices = load_random_noise(type_, n_samples,
                                                         self.lims, self.fs)
            else:
                raise ValueError(('type_ must start with noise_ or '
                                  'dcase_, got %s' % type_))
            return x, filepath, indices
