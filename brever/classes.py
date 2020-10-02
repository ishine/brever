import numpy as np
import scipy.signal
import random
import re

from .utils import pca, frame, rms
from .filters import mel_filterbank, gammatone_filterbank
from .mixture import Mixture, colored_noise, add_decay
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


class UnitRMSScaler:
    def __init__(self, active=True):
        self.active = active
        self.gain = None

    def fit(self, signal):
        rms_max = rms(signal).max()
        self.gain = 1/rms_max

    def scale(self, signal):
        if self.active:
            return self.gain*signal
        else:
            return signal


class Filterbank:
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

    def rfilt(self, x_filt):
        x = np.zeros((len(x_filt), x_filt.shape[2]))
        for i in range(self.n_filters):
            x += scipy.signal.lfilter(self.b[i], self.a[i], x_filt[::-1, i, :],
                                      axis=0)
        return x[::-1].squeeze()


class Framer:
    def __init__(self, frame_length, hop_length, window, center):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = window
        self.center = center

    def frame(self, x):
        return frame(x, frame_length=self.frame_length,
                     hop_length=self.hop_length, window=self.window,
                     center=self.center)


class FeatureExtractor:
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


class LabelExtractor:
    def __init__(self, label):
        self.label = label

    def run(self, target, noise):
        label_func = getattr(labels_module, self.label)
        return label_func(target, noise, filtered=True, framed=True)


class RandomMixtureMaker:
    def __init__(self, fs, rooms, target_angles, target_snrs,
                 directional_noise_numbers, directional_noise_types,
                 directional_noise_angles, directional_noise_snrs,
                 diffuse_noise_on, diffuse_noise_color, diffuse_noise_ltas_eq,
                 mixture_pad, mixture_rb, mixture_rms_jitter, path_timit,
                 path_surrey, path_dcase, filelims_target,
                 filelims_directional_noise, decay_on, decay_color,
                 decay_rt60s, decay_drrs, decay_delays):
        self.fs = fs
        self.rooms = rooms
        self.target_angles = target_angles
        self.target_snrs = target_snrs
        self.directional_noise_snrs = directional_noise_snrs
        self.directional_noise_numbers = directional_noise_numbers
        self.directional_noise_types = directional_noise_types
        self.directional_noise_angles = directional_noise_angles
        self.diffuse_noise_on = diffuse_noise_on
        self.diffuse_noise_color = diffuse_noise_color
        self.diffuse_noise_ltas_eq = diffuse_noise_ltas_eq
        self.mixture_pad = mixture_pad
        self.mixture_rb = mixture_rb
        self.mixture_rms_jitter = mixture_rms_jitter
        self.path_timit = path_timit
        self.path_surrey = path_surrey
        self.path_dcase = path_dcase
        self.filelims_target = filelims_target
        self.filelims_directional_noise = filelims_directional_noise
        self.decay_on = decay_on
        self.decay_color = decay_color
        self.decay_rt60s = decay_rt60s
        self.decay_drrs = decay_drrs
        self.decay_delays = decay_delays

    def make(self):
        self.mixture = Mixture()
        self.metadata = {}
        room = self.get_random_room()
        decayer = self.get_random_decayer()
        self.add_random_target(room, decayer)
        self.add_random_directional_noises(room, decayer)
        self.add_random_diffuse_noise(room)
        self.set_random_dir_to_diff_snr()
        self.set_random_target_snr()
        self.set_random_rms()
        return self.mixture, self.metadata

    def get_random_room(self):
        room = choice(self.rooms)
        self.metadata['room'] = room
        return room

    def get_random_decayer(self):
        rt60 = choice(self.decay_rt60s)
        drr = choice(self.decay_drrs)
        delay = choice(self.decay_delays)
        decayer = Decayer(
            rt60,
            drr,
            delay,
            self.fs,
            self.decay_color,
            self.decay_on,
        )
        if self.decay_on:
            self.metadata['decay'] = {}
            self.metadata['decay']['rt60'] = rt60
            self.metadata['decay']['drr'] = drr
            self.metadata['decay']['delay'] = delay
            self.metadata['decay']['color'] = self.decay_color
        return decayer

    def add_random_target(self, room, decayer):
        angle = choice(self.target_angles)
        brir = self._load_brirs(room, angle)
        brir = decayer.run(brir)
        target, filename = load_random_target(
            self.path_timit,
            self.filelims_target,
            self.fs
        )
        self.mixture.add_target(
            x=target,
            brir=brir,
            rb=self.mixture_rb,
            pad=self.mixture_pad,
            fs=self.fs,
        )
        self.metadata['target'] = {}
        self.metadata['target']['angle'] = angle
        self.metadata['target']['filename'] = filename

    def add_random_directional_noises(self, room, decayer):
        number = choice(self.directional_noise_numbers)
        types = choices(self.directional_noise_types, k=number)
        angles = choices(self.directional_noise_angles, k=number)
        noises, files, indices = self._load_noises(types, len(self.mixture))
        brirs = self._load_brirs(room, angles)
        brirs = [decayer.run(brir) for brir in brirs]
        self.mixture.add_directional_noises(noises, brirs)
        self.metadata['directional'] = {}
        self.metadata['directional']['number'] = number
        self.metadata['directional']['sources'] = []
        for i in range(number):
            source_metadata = {
                'angle': angles[i],
                'type': types[i],
                'filename': files[i],
                'indices': indices[i],
            }
            self.metadata['directional']['sources'].append(source_metadata)

    def add_random_diffuse_noise(self, room):
        if self.diffuse_noise_on:
            brirs = self._load_brirs(room)
            self.mixture.add_diffuse_noise(
                brirs,
                self.diffuse_noise_color,
                self.diffuse_noise_ltas_eq,
            )
            self.metadata['diffuse'] = {}
            self.metadata['diffuse']['color'] = self.diffuse_noise_color
            self.metadata['diffuse']['ltas_eq'] = self.diffuse_noise_ltas_eq

    def set_random_dir_to_diff_snr(self):
        snr = choice(self.directional_noise_snrs)
        if self.metadata['directional']['number'] == 0:
            return
        if self.diffuse_noise_on:
            self.mixture.adjust_dir_to_diff_snr(snr)
            self.metadata['directional']['snr'] = snr

    def set_random_target_snr(self):
        snr = choice(self.target_snrs)
        if self.metadata['directional']['number'] == 0:
            if not self.diffuse_noise_on:
                return
        self.mixture.adjust_target_snr(snr)
        self.metadata['target']['snr'] = snr

    def set_random_rms(self):
        rms_dB = choice(self.mixture_rms_jitter)
        self.mixture.adjust_rms(rms_dB)
        self.metadata['rms_dB'] = rms_dB

    def _load_brirs(self, room, angles=None):
        if angles is None or isinstance(angles, list):
            brirs, fs = load_brirs(self.path_surrey, room, angles)
            if fs is not None and fs != self.fs:
                raise ValueError(('the brir samplerate obtained from '
                                  'load_brirs(%s, %s) does not match the '
                                  'RandomMixtureMaker instance samplerate '
                                  'attribute (%i vs %i)'
                                  % (room, angles, fs, self.fs)))
            return brirs
        else:
            brir, fs = load_brir(self.path_surrey, room, angles)
            if fs is not None and fs != self.fs:
                raise ValueError(('the brir samplerate obtained from '
                                  'load_brir(%s, %s) does not match the '
                                  'RandomMixtureMaker instance samplerate '
                                  'attribute (%i vs %i)'
                                  % (room, angles, fs, self.fs)))
            return brir

    def _load_noises(self, types, n_samples):
        if isinstance(types, list):
            if not types:
                return [], [], []
            zipped = [self._load_noises(type_, n_samples) for type_ in types]
            xs, filepaths, indicess = zip(*zipped)
            return xs, filepaths, indicess
        else:
            type_ = types
            if type_ is None:
                x, filepath, indices = None, None, None
            elif type_.startswith('noise_'):
                color = re.match('^noise_(.*)$', type_).group(1)
                x = colored_noise(color, n_samples)
                filepath = None
                indices = None
            elif type_.startswith('dcase_'):
                x, filepath, indices = load_random_noise(
                    self.path_dcase,
                    type_,
                    n_samples,
                    self.filelims_directional_noise,
                    self.fs,
                )
            else:
                raise ValueError(('type_ must start with noise_ or '
                                  'dcase_, got %s' % type_))
            return x, filepath, indices


class Decayer:
    def __init__(self, rt60, drr, delay, fs, color, active):
        self.rt60 = rt60
        self.drr = drr
        self.delay = delay
        self.fs = fs
        self.color = color
        self.active = active

    def run(self, brir):
        if self.active:
            brir = add_decay(
                brir,
                self.rt60,
                self.drr,
                self.delay,
                self.fs,
                self.color,
            )
        return brir


def choice(sequence):
    if isinstance(sequence, set):
        return random.choice(list(sequence))
    return random.choice(sequence)


def choices(sequence, k):
    if isinstance(sequence, set):
        return random.choices(list(sequence), k=k)
    return random.choices(sequence, k=k)
