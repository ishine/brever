import queue
import threading
import numpy as np
from scipy.signal import lfilter, sosfilt
import random
import re
from functools import partial
import inspect
import logging

from .utils import pca, frame, rms
from .filters import mel_filterbank, gammatone_filterbank
from .mixture import Mixture, colored_noise, add_decay, match_ltas
from .io import (load_random_target, load_brirs, load_random_noise,
                 get_available_angles, get_rooms, get_average_duration,
                 get_all_filepaths, get_ltas)
from .config import defaults
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
    def __init__(self, kind, n_filters, f_min, f_max, fs, order, output='ba'):
        if output not in ['ba', 'sos']:
            raise ValueError('only "ba" and "sos" outputs are not supported, '
                             f'got "{output}"')
        if output != 'ba' and kind == 'gammatone':
            raise ValueError('only "ba" output is supported for gammatone '
                             f'filterbank, got "{output}"')
        self.kind = kind
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.fs = fs
        self.order = order
        self.output = output
        if kind == 'mel':
            self.filters, self.fc = mel_filterbank(n_filters, f_min, f_max, fs,
                                                   order, output)
        elif kind == 'gammatone':
            self.filters, self.fc = gammatone_filterbank(n_filters, f_min,
                                                         f_max, fs, order)
        else:
            raise ValueError('filtertype must be "mel" or "gammatone", got '
                             f'"{kind}"')

    def filt(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_samples, n_channels = x.shape
        x_filt = np.zeros((n_samples, self.n_filters, n_channels))
        for i in range(self.n_filters):
            if self.output == 'ba':
                filter_func = partial(lfilter, self.filters[i][0],
                                      self.filters[i][1], axis=0)
            elif self.output == 'sos':
                filter_func = partial(sosfilt, self.filters[i], axis=0)
            x_filt[:, i, :] = filter_func(x)
        return x_filt.squeeze()

    def rfilt(self, x_filt):
        if x_filt.ndim == 2:
            x_filt = x_filt[:, :, np.newaxis]
        x = np.zeros((len(x_filt), x_filt.shape[2]))
        for i in range(self.n_filters):
            if self.output == 'ba':
                filter_func = partial(lfilter, self.filters[i][0],
                                      self.filters[i][1], axis=0)
            elif self.output == 'sos':
                filter_func = partial(sosfilt, self.filters[i], axis=0)
            else:
                raise ValueError('wrong output type')
            x += filter_func(x_filt[::-1, i, :])
        return x[::-1].squeeze()


class MultiThreadFilterbank(Filterbank):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _thread_target(self, q, filter_, x, i):
        if self.output == 'ba':
            q.put((i, lfilter(filter_[0], filter_[1], x, axis=0)))
        elif self.output == 'sos':
            q.put((i, sosfilt(filter_, x, axis=0)))
        else:
            raise ValueError('wrong output type')

    def filt(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        q = queue.Queue()

        n_samples, n_channels = x.shape
        for i in range(self.n_filters):
            t = threading.Thread(
                target=self._thread_target,
                args=(
                    q,
                    self.filters[i],
                    x,
                    i,
                ),
            )
            t.daemon = True
            t.start()

        x_filt = np.zeros((n_samples, self.n_filters, n_channels))
        for j in range(self.n_filters):
            i, data = q.get()
            x_filt[:, i, :] = data

        return x_filt.squeeze()

    def rfilt(self, x_filt):
        if x_filt.ndim == 2:
            x_filt = x_filt[:, :, np.newaxis]

        q = queue.Queue()

        for i in range(self.n_filters):
            t = threading.Thread(
                target=self._thread_target,
                args=(
                    q,
                    self.filters[i],
                    x_filt[::-1, i, :],
                    i
                ),
            )
            t.daemon = True
            t.start()

        x = np.zeros((len(x_filt), x_filt.shape[2]))
        for j in range(self.n_filters):
            i, data = q.get()
            x += data

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
        # pre-compute energy
        x = x.mean(axis=-1)  # average channels
        energy = x**2  # get energy
        energy = energy.mean(axis=1)  # average each frame
        # main loop
        output = []
        for feature in self.features:
            feature_func = getattr(features_module, feature)
            argspec = inspect.getfullargspec(feature_func)
            if 'energy' in argspec.args:
                featmat = feature_func(x, filtered=True, framed=True,
                                       energy=energy)
            else:
                featmat = feature_func(x, filtered=True, framed=True)
            output.append(featmat)
        self.indices = []
        i_start = 0
        for feature_set in output:
            i_end = i_start + feature_set.shape[1]
            self.indices.append((i_start, i_end))
            i_start = i_end
        return np.hstack(output)


class LabelExtractor:
    def __init__(self, labels):
        self.labels = labels
        self.indices = None

    def run(self, mix_object):
        output = []
        for label in self.labels:
            label_func = getattr(labels_module, label)
            output.append(label_func(mix_object, filtered=True, framed=True))
        self.indices = []
        i_start = 0
        for label_set in output:
            i_end = i_start + label_set.shape[1]
            self.indices.append((i_start, i_end))
            i_start = i_end
        return np.hstack(output)


class RandomPool:
    def __init__(self, pool, seed=None, weights=None):
        self.set_pool(pool, weights)
        self.random = random.Random(seed)

    def get(self):
        item, = self.random.choices(self.pool, weights=self.weights)
        return item

    def set_pool(self, pool, weights=None):
        self.weights = weights
        if isinstance(pool, set):
            self.pool = sorted(pool)
            if weights is not None:
                if not isinstance(weights, dict):
                    raise ValueError('weights must be dict when pool is set')
                if set(weights.keys()) != pool:
                    raise ValueError('weights keys do not match pool')
                self.weights = [weights[x] for x in pool]
        else:
            self.pool = pool
            if weights is not None:
                if not isinstance(weights, list):
                    raise ValueError('weights must be list when pool is list')
                if len(weights) != len(pool):
                    raise ValueError('weights and pool must have same length')
                self.weights = weights


class MultiRandomPool:
    def __init__(self, pool, k_max, seed=None):
        self.set_pool(pool)
        if seed is None:
            self.randoms = [random.Random() for i in range(k_max)]
        else:
            self.randoms = [random.Random(seed+i) for i in range(k_max)]
        self.k_max = k_max

    def get(self, k):
        assert k <= self.k_max
        output = [self.randoms[i].choice(self.pool) for i in range(self.k_max)]
        return output[:k]

    def set_pool(self, pool):
        if isinstance(pool, set):
            self.pool = sorted(pool)
        else:
            self.pool = pool


class Seeder:
    def __init__(self, seed_value, max_seed):
        self.random = random.Random(seed_value)
        self.max_seed = max_seed

    def get(self):
        return self.random.randrange(self.max_seed)


class ContinuousRandomGenerator:
    def __init__(self, dist_name, dist_args, seed=None):
        self.random = np.random.RandomState(seed)
        self.dist_name = dist_name
        self.dist_args = dist_args

    def get(self):
        dist_func = getattr(self.random, self.dist_name)
        return dist_func(*self.dist_args)


class RandomMixtureMaker:
    def __init__(self, fs, rooms, speakers,
                 target_snr_dist_name, target_snr_dist_args, target_angle_min,
                 target_angle_max, dir_noise_nums, dir_noise_types,
                 dir_noise_snrs, dir_noise_angle_min, dir_noise_angle_max,
                 diffuse_noise_on, diffuse_noise_color, diffuse_noise_ltas_eq,
                 mixture_pad, mixture_rb, mixture_rms_jitter_on,
                 mixture_rms_jitters, filelims_target, filelims_dir_noise,
                 filelims_room, decay_on, decay_color, decay_rt60s,
                 decay_drr_dist_name, decay_drr_dist_args, decay_delays,
                 seed_on, seed_value, uniform_tmr):

        if not seed_on:
            seed_value = None
        seeder = Seeder(seed_value, 100)

        self.def_cfg = defaults()
        self.fs = fs
        self.room_regexps = RandomPool(rooms, seeder.get())
        self.rooms = RandomPool([], seeder.get())

        # For the speakers random generator, the probability distribution must
        # be weighted according to the average duration of the sentences,
        # otherwise the speech material in the dataset will be unbalanced.
        # Example: TIMIT sentences are 3 seconds long on average, while LIBRI
        # sentences are 12 seconds long on average, so making a dataset using
        # 50 TIMIT sentences and 50 LIBRI sentences will result in much more
        # LIBRI samples.
        weights = {speaker: 1/get_average_duration(speaker, self.def_cfg)
                   for speaker in speakers}
        self.speakers = RandomPool(speakers, seeder.get(), weights)

        self.target_snrs = ContinuousRandomGenerator(target_snr_dist_name,
                                                     target_snr_dist_args,
                                                     seeder.get())
        self.target_angle_min = target_angle_min
        self.target_angle_max = target_angle_max
        self.dir_noise_snrs = RandomPool(dir_noise_snrs, seeder.get())
        self.dir_noise_nums = RandomPool(dir_noise_nums, seeder.get())
        self.dir_noise_types = MultiRandomPool(dir_noise_types,
                                               max(dir_noise_nums),
                                               seeder.get())
        self.dir_noise_angle_min = dir_noise_angle_min
        self.dir_noise_angle_max = dir_noise_angle_max
        self.diffuse_noise_on = diffuse_noise_on
        self.diffuse_noise_color = diffuse_noise_color
        self.diffuse_noise_ltas_eq = diffuse_noise_ltas_eq
        self.mixture_pad = mixture_pad
        self.mixture_rb = mixture_rb
        self.mixture_rms_jitter_on = mixture_rms_jitter_on
        self.mixture_rms_jitters = RandomPool(mixture_rms_jitters,
                                              seeder.get())
        self.filelims_target = filelims_target
        self.filelims_dir_noise = filelims_dir_noise
        self.filelims_room = filelims_room
        self.decay_on = decay_on
        self.decay_color = decay_color
        self.decay_rt60s = RandomPool(decay_rt60s, seeder.get())
        self.decay_drrs = ContinuousRandomGenerator(decay_drr_dist_name,
                                                    decay_drr_dist_args,
                                                    seeder.get())
        self.decay_delays = RandomPool(decay_delays, seeder.get())

        self.target_filename_randomizer = RandomPool([], seeder.get())
        self.noise_filename_randomizer = MultiRandomPool([],
                                                         max(dir_noise_nums),
                                                         seeder.get())
        self.uniform_tmr = uniform_tmr
        self.tmrs = ContinuousRandomGenerator('uniform', [], seeder.get())
        self.target_angles = RandomPool([], seeder.get())
        self.dir_noise_angles = MultiRandomPool([], max(dir_noise_nums),
                                                seeder.get())

        self.bbl_speakers = RandomPool(speakers, seeder.get(), weights)
        self.bbl_filename_randomizer = RandomPool([], seeder.get())

        self.all_filepaths_dict = {
            speaker: get_all_filepaths(speaker, def_cfg=self.def_cfg)
            for speaker in speakers
        }

        if ((diffuse_noise_on and diffuse_noise_ltas_eq)
                or ('ssn' in dir_noise_types and max(dir_noise_nums) > 0)):
            merged_filepaths = [f for l_ in self.all_filepaths_dict.values()
                                for f in l_]
            logging.info(f'Calculating LTAS from {len(merged_filepaths)} '
                         'speech files')
            self.ltas = get_ltas(all_filepaths=merged_filepaths,
                                 def_cfg=self.def_cfg)
        else:
            self.ltas = None

    def make(self):
        mixture = Mixture()
        metadata = {}
        room, metadata = self.get_random_room(metadata)
        decayer, metadata = self.get_random_decayer(metadata)
        mixture, metadata = self.add_random_target(mixture, metadata, room,
                                                   decayer)
        mixture, metadata = self.add_random_dir_noises(mixture, metadata, room,
                                                       decayer)
        mixture, metadata = self.add_random_diffuse_noise(mixture, metadata,
                                                          room)
        mixture, metadata = self.set_random_dir_to_diff_snr(mixture, metadata)
        mixture, metadata = self.set_random_target_snr(mixture, metadata)
        mixture, metadata = self.set_random_tmr(mixture, metadata)
        mixture, metadata = self.set_random_rms(mixture, metadata)
        metadata = self.get_long_term_labels(mixture, metadata)
        return mixture, metadata

    def get_random_room(self, metadata):
        room_regexp = self.room_regexps.get()
        self.rooms.set_pool(get_rooms(room_regexp))
        room = self.rooms.get()
        metadata['room'] = room
        return room, metadata

    def get_random_decayer(self, metadata):
        rt60 = self.decay_rt60s.get()
        drr = self.decay_drrs.get()
        delay = self.decay_delays.get()
        decayer = Decayer(
            rt60,
            drr,
            delay,
            self.fs,
            self.decay_color,
            self.decay_on,
        )
        if self.decay_on:
            metadata['decay'] = {}
            metadata['decay']['rt60'] = rt60
            metadata['decay']['drr'] = drr
            metadata['decay']['delay'] = delay
            metadata['decay']['color'] = self.decay_color
        return decayer, metadata

    def add_random_target(self, mixture, metadata, room, decayer):
        angles = get_available_angles(
            room,
            def_cfg=self.def_cfg,
            angle_min=self.target_angle_min,
            angle_max=self.target_angle_max,
            parity=self.filelims_room,
        )
        self.target_angles.set_pool(angles)
        angle = self.target_angles.get()
        brir = self._load_brirs(room, angle)
        brir = decayer.run(brir)
        speaker = self.speakers.get()
        target, filename = load_random_target(
            speaker,
            self.filelims_target,
            self.fs,
            randomizer=self.target_filename_randomizer.random,
            def_cfg=self.def_cfg,
            all_filepaths=self.all_filepaths_dict[speaker]
        )
        mixture.add_target(
            x=target,
            brir=brir,
            rb=self.mixture_rb,
            t_pad=self.mixture_pad,
            fs=self.fs,
        )
        metadata['target'] = {}
        metadata['target']['angle'] = angle
        metadata['target']['filename'] = filename
        return mixture, metadata

    def add_random_dir_noises(self, mixture, metadata, room, decayer):
        number = self.dir_noise_nums.get()
        types = self.dir_noise_types.get(number)

        if types and types[0] == 'bbl':

            angles = get_available_angles(room, self.def_cfg)
            if len(angles) == 1:
                raise ValueError('cannot use bbl noise with a room that only '
                                 'has one brir')
            angles = [a for a in angles if a != metadata['target']['angle']]
            brirs = self._load_brirs(room, angles)
            brirs = [decayer.run(brir) for brir in brirs]
            noises = []
            for i in range(len(angles)):
                speaker = self.bbl_speakers.get()
                noise, _ = load_random_target(
                    speaker,
                    self.filelims_target,
                    self.fs,
                    randomizer=self.bbl_filename_randomizer.random,
                    def_cfg=self.def_cfg,
                    all_filepaths=self.all_filepaths_dict[speaker]
                )
                while len(noise) < len(mixture):
                    noise_, _ = load_random_target(
                        speaker,
                        self.filelims_target,
                        self.fs,
                        randomizer=self.bbl_filename_randomizer.random,
                        def_cfg=self.def_cfg,
                        all_filepaths=self.all_filepaths_dict[speaker]
                    )
                    noise = np.hstack((noise, noise_))
                noise = noise[:len(mixture)]
                noises.append(noise)
            mixture.add_dir_noises(noises, brirs)
            metadata['directional'] = {}
            metadata['directional']['number'] = len(angles)
            metadata['directional']['sources'] = []

        else:

            types = [t for t in types if t != 'bbl']
            number = len(types)
            angles = get_available_angles(
                room,
                def_cfg=self.def_cfg,
                angle_min=self.dir_noise_angle_min,
                angle_max=self.dir_noise_angle_max,
                parity=self.filelims_room,
            )
            if len(angles) > 1:
                angles = [a for a in angles if a !=
                          metadata['target']['angle']]
            self.dir_noise_angles.set_pool(angles)
            angles = self.dir_noise_angles.get(number)
            noises, files, indices = self._load_noises(
                types,
                len(mixture),
            )
            brirs = self._load_brirs(room, angles)
            brirs = [decayer.run(brir) for brir in brirs]
            mixture.add_dir_noises(noises, brirs)
            metadata['directional'] = {}
            metadata['directional']['number'] = number
            metadata['directional']['sources'] = []
            for i in range(number):
                source_metadata = {
                    'angle': angles[i],
                    'type': types[i],
                    'filename': files[i],
                    'indices': indices[i],
                }
                metadata['directional']['sources'].append(source_metadata)

        return mixture, metadata

    def add_random_diffuse_noise(self, mixture, metadata, room):
        if self.diffuse_noise_on:
            brirs = self._load_brirs(room)
            mixture.add_diffuse_noise(
                brirs,
                self.diffuse_noise_color,
                self.ltas,
            )
            metadata['diffuse'] = {}
            metadata['diffuse']['color'] = self.diffuse_noise_color
            metadata['diffuse']['ltas_eq'] = self.diffuse_noise_ltas_eq
        return mixture, metadata

    def set_random_dir_to_diff_snr(self, mixture, metadata):
        snr = self.dir_noise_snrs.get()
        if metadata['directional']['number'] == 0:
            return mixture, metadata
        if self.diffuse_noise_on:
            mixture.adjust_dir_to_diff_snr(snr)
            metadata['directional']['snr'] = snr
        return mixture, metadata

    def set_random_target_snr(self, mixture, metadata):
        snr = self.target_snrs.get()
        if metadata['directional']['number'] == 0:
            if not self.diffuse_noise_on:
                return mixture, metadata
        mixture.adjust_target_snr(snr)
        metadata['target']['snr'] = snr
        return mixture, metadata

    def set_random_tmr(self, mixture, metadata):
        tmr = self.tmrs.get()
        alpha = self.tmrs.get()
        if self.uniform_tmr:
            target_energy = np.sum(mixture.early_target.mean(axis=1)**2)
            new_masker_energy = target_energy*(1/tmr-1)
            new_noise_energy = alpha*new_masker_energy
            new_reverb_energy = (1-alpha)*new_masker_energy
            current_noise_energy = np.sum(mixture.noise.mean(axis=1)**2)
            current_reverb_energy = np.sum(mixture.late_target.mean(axis=1)**2)
            noise_gain = (new_noise_energy/current_noise_energy)**0.5
            reverb_gain = (new_reverb_energy/current_reverb_energy)**0.5
            if mixture.dir_noise is not None:
                mixture.dir_noise = noise_gain*mixture.dir_noise
            if mixture.diffuse_noise is not None:
                mixture.diffuse_noise = noise_gain*mixture.diffuse_noise
            mixture.late_target = reverb_gain*mixture.late_target
            metadata['uniform_tmr'] = {}
            metadata['uniform_tmr']['tmr'] = tmr
            metadata['uniform_tmr']['alpha'] = alpha
            metadata['target']['snr'] = None
            metadata['decay']['drr'] = None
        return mixture, metadata

    def set_random_rms(self, mixture, metadata):
        rms_dB = self.mixture_rms_jitters.get()
        if self.mixture_rms_jitter_on:
            mixture.adjust_rms(rms_dB)
            metadata['rms_dB'] = rms_dB
        return mixture, metadata

    def get_long_term_labels(self, mixture, metadata):
        metadata['lt_labels'] = {}
        for label in ['tmr', 'tnr', 'trr']:
            metadata['lt_labels'][label] = mixture.get_long_term_label(label)
        return metadata

    def _load_brirs(self, room, angles=None):
        brirs, _ = load_brirs(room, angles, self.fs, self.def_cfg)
        return brirs

    def _load_noises(self, types, n_samples):
        if not types:
            return [], [], []
        zipped = []
        randomizers = self.noise_filename_randomizer.randoms
        for type_, randomizer in zip(types, randomizers):
            if type_ is None:
                x, filepath, indices = None, None, None
            elif type_.startswith('colored_'):
                color = re.match('^colored_(.*)$', type_).group(1)
                x = colored_noise(color, n_samples)
                filepath = None
                indices = None
            elif type_ == 'ssn':
                x = colored_noise('white', n_samples)
                x = match_ltas(x, self.ltas)
                filepath = None
                indices = None
            else:
                x, filepath, indices = load_random_noise(
                    type_,
                    n_samples,
                    self.filelims_dir_noise,
                    self.fs,
                    randomizer=randomizer,
                    def_cfg=self.def_cfg,
                )
            zipped.append((x, filepath, indices))
        xs, filepaths, indicess = zip(*zipped)
        return xs, filepaths, indicess


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


class DefaultRandomMixtureMaker(RandomMixtureMaker):
    def __init__(self):
        config = defaults()
        super().__init__(
            fs=config.PRE.FS,
            rooms=config.PRE.MIX.RANDOM.ROOMS,
            target_snrs=range(
                config.PRE.MIX.RANDOM.TARGET.SNR.MIN,
                config.PRE.MIX.RANDOM.TARGET.SNR.MAX + 1,
            ),
            dir_noise_nums=range(
                config.PRE.MIX.RANDOM.SOURCES.NUMBER.MIN,
                config.PRE.MIX.RANDOM.SOURCES.NUMBER.MAX + 1,
            ),
            dir_noise_types=config.PRE.MIX.RANDOM.SOURCES.TYPES,
            dir_noise_snrs=range(
                config.PRE.MIX.RANDOM.SOURCES.SNR.MIN,
                config.PRE.MIX.RANDOM.SOURCES.SNR.MAX + 1,
            ),
            diffuse_noise_on=config.PRE.MIX.DIFFUSE.ON,
            diffuse_noise_color=config.PRE.MIX.DIFFUSE.COLOR,
            diffuse_noise_ltas_eq=config.PRE.MIX.DIFFUSE.LTASEQ,
            mixture_pad=config.PRE.MIX.PADDING,
            mixture_rb=config.PRE.MIX.REFLECTIONBOUNDARY,
            mixture_rms_jitter_on=config.PRE.MIX.RANDOM.RMSDB.ON,
            mixture_rms_jitters=range(
                config.PRE.MIX.RANDOM.RMSDB.MIN,
                config.PRE.MIX.RANDOM.RMSDB.MAX + 1,
            ),
            filelims_dir_noise=config.PRE.MIX.FILELIMITS.NOISE,
            filelims_target=config.PRE.MIX.FILELIMITS.TARGET,
            decay_on=config.PRE.MIX.DECAY.ON,
            decay_color=config.PRE.MIX.DECAY.COLOR,
            decay_rt60s=np.arange(
                config.PRE.MIX.RANDOM.DECAY.RT60.MIN,
                config.PRE.MIX.RANDOM.DECAY.RT60.MAX,
                config.PRE.MIX.RANDOM.DECAY.RT60.STEP,
                dtype=float,
            ),
            decay_drrs=np.arange(
                config.PRE.MIX.RANDOM.DECAY.DRR.MIN,
                config.PRE.MIX.RANDOM.DECAY.DRR.MAX,
                config.PRE.MIX.RANDOM.DECAY.DRR.STEP,
                dtype=float,
            ),
            decay_delays=np.arange(
                config.PRE.MIX.RANDOM.DECAY.DELAY.MIN,
                config.PRE.MIX.RANDOM.DECAY.DELAY.MAX,
                config.PRE.MIX.RANDOM.DECAY.DELAY.STEP,
                dtype=float,
            ),
        )


class DefaultFilterbank(Filterbank):
    def __init__(self):
        config = defaults()
        super().__init__(
            kind=config.PRE.FILTERBANK.KIND,
            n_filters=config.PRE.FILTERBANK.NFILTERS,
            f_min=config.PRE.FILTERBANK.FMIN,
            f_max=config.PRE.FILTERBANK.FMAX,
            fs=config.PRE.FS,
            order=config.PRE.FILTERBANK.ORDER,
        )


class DefaultFramer(Framer):
    def __init__(self):
        config = defaults()
        super().__init__(
            frame_length=config.PRE.FRAMER.FRAMELENGTH,
            hop_length=config.PRE.FRAMER.HOPLENGTH,
            window=config.PRE.FRAMER.WINDOW,
            center=config.PRE.FRAMER.CENTER,
        )
