import logging
import random
import re

import numpy as np
import scipy.signal

from .config import defaults
from .io import (load_random_target, load_brirs, load_random_noise,
                 get_available_angles, get_rooms, get_average_duration,
                 get_all_filepaths, get_ltas)
from .utils import pad, fft_freqs, rms


def spatialize(x, brir):
    """
    Signal spatialization.

    Spatializes a single channel input audio signal with the provided BRIR
    using overlap-add convolution method. The last samples of the output
    signal are discarded such that it matches the length of the input signal.

    Parameters
    ----------
    x : array_like
        Monaural audio signal to spatialize.
    brir : array_like
        Binaural room impulse response. Shape `(len(brir), 2)`.

    Returns
    -------
    y: array_like
        Binaural audio signal. Shape `(len(x), 2)`.
    """
    n = len(x)
    x_left = scipy.signal.oaconvolve(x, brir[:, 0], mode='full')[:n]
    x_right = scipy.signal.oaconvolve(x, brir[:, 1], mode='full')[:n]
    return np.vstack([x_left, x_right]).T


def colored_noise(color, n_samples):
    """
    Colored noise.

    Generates noise with power spectral density following a `1/f**alpha`
    distribution.

    Parameters
    ----------
    color : {'brown', 'pink', 'white', 'blue', 'violet'}
        Color of the noise.
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    x: array_like
        Colored noise. One-dimensional with length `n_samples`.
    """
    colors = {
        'brown': 2,
        'pink': 1,
        'white': 0,
        'blue': -1,
        'violet': -2,
    }
    if color not in colors.keys():
        raise ValueError('color must be either one of %s' % colors.keys())
    alpha = colors[color]
    scaling = fft_freqs(fs=1, n_fft=n_samples)
    scaling[0] = scaling[1]
    scaling **= -alpha/2
    x = np.random.randn(n_samples)
    X = np.fft.rfft(x)
    X *= scaling
    x = np.fft.irfft(X, n_samples).real
    return x


def match_ltas(x, ltas, n_fft=512, hop_length=256):
    """
    Long-term-average-spectrum (LTAS) matching.

    Filters the input signal in the short-time Fourier transform (STFT) domain
    such that it presents a specific long-term-average-spectrum (LTAS). The
    considered LTAS is the average LTAS across channels

    Parameters
    ----------
    x : array_like
        Input signal. Shape `(n_samples, n_channels)`.
    ltas : array_like
        Desired long-term-average-spectrum (LTAS). One-dimensional with
        length `n_fft`.
    n_fft : int, optional
        Number of FFT points. Default is 512.
    hop_length : int, optional
        Frame shift in samples. Default is 256.

    Returns
    -------
    y : array_like
        Output signal with LTAS equal to `ltas`.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        flat_output = True
    else:
        flat_output = False
    n = len(x)
    noverlap = n_fft-hop_length
    _, _, X = scipy.signal.stft(x, nperseg=n_fft, noverlap=noverlap, axis=0)
    ltas_X = np.mean(np.abs(X**2), axis=(1, 2))
    EQ = (ltas/ltas_X)**0.5
    X *= EQ[:, np.newaxis, np.newaxis]
    _, x = scipy.signal.istft(X, nperseg=n_fft, noverlap=noverlap, freq_axis=0)
    x = x.T
    if flat_output:
        x = x.ravel()
    return x[:n]


def split_brir(brir, reflection_boundary=50e-3, fs=16e3, max_itd=1e-3):
    """
    BRIR split.

    Splits a BRIR into a direct or early reflections component and a reverb or
    late reflections component according to a reflection boundary.

    Parameters
    ----------
    brir : array_like
        Input BRIR. Shape `(len(brir), 2)`.
    reflection_boundary : float, optional
        Reflection boundary in seconds, This is the limit between early and
        late reflections. Default is 50e-3.
    fs : float or int, optional
        Sampling frequency. Default is 16e3.
    max_itd : float, optional
        Maximum interaural time difference in seconds. Used to compare the
        location of the highest peak in each channel and adjust if necessary.
        Default is 1e-3.

    Returns
    -------
    brir_early: array_like
        Early reflections part of input BRIR. Same shape as `brir`.
    brir_late: array_like
        Late reflections part of input BRIR. Same shape as `brir`.
    """
    peak_i = np.argmax(np.abs(brir), axis=0)
    peak_val = np.max(np.abs(brir), axis=0)
    max_delay = round(max_itd*fs)
    if peak_val[0] > peak_val[1]:
        segment = np.abs(brir[peak_i[0]:peak_i[0]+max_delay, 1])
        peak_i[1] = peak_i[0] + np.argmax(segment)
    else:
        segment = np.abs(brir[peak_i[1]:peak_i[1]+max_delay, 0])
        peak_i[0] = peak_i[1] + np.argmax(segment)
    win_early = np.zeros(brir.shape)
    win_early[:peak_i[0] + round(reflection_boundary*fs), 0] = 1
    win_early[:peak_i[1] + round(reflection_boundary*fs), 1] = 1
    win_late = 1 - win_early
    brir_early = win_early*brir
    brir_late = win_late*brir
    return brir_early, brir_late


def adjust_snr(signal, noise, snr, slice_=None):
    """
    SNR adjustement.

    Scales a noise signal given a target signal and a desired signal-to-noise
    ratio (SNR). The average energy that is used to calculate the SNR is the
    one of the monaural average signal across channels.

    Parameters
    ----------
    signal: array_like
        Target signal. Shape `(n_samples, n_channels)`.
    noise: array
        Noise signal. Shape `(n_samples, n_channels)`.
    snr:
        Desired SNR.
    slice_: slice or None, optional
        Slice of the target and noise signals from which the SNR should be
        calculated. Default is `None`, which means the energy of the entire
        signals are calculated.

    Returns
    -------
    noise_scaled: array_like
        Scaled noise. The SNR between the target signal and the new scaled
        noise is equal to `snr`.
    """
    if slice_ is None:
        slice_ = np.s_[:]
    energy_signal = np.sum(signal[slice_].mean(axis=1)**2)
    energy_noise = np.sum(noise[slice_].mean(axis=1)**2)
    if energy_signal == 0:
        raise ValueError("Can't scale noise signal if target signal is 0!")
    if energy_noise == 0:
        raise ValueError("Can't scale noise signal if it equals 0!")
    gain = (10**(-snr/10)*energy_signal/energy_noise)**0.5
    noise_scaled = gain*noise
    return noise_scaled, gain


def adjust_rms(signal, rms_dB):
    """
    RMS adjustment.

    Scales a signal such that it presents a given root-mean-square (RMS) in dB.
    Note that the reference value in the dB calculation is 1, meaning a
    signal with an RMS of 0 dB has an absolute RMS of 1. In the case of white
    noise, this leads to a signal with unit variance, with values likely to be
    outside the [-1, 1] range and cause clipping.

    Parameters
    ----------
    signal:
        Input signal.
    rms_dB:
        Desired RMS in dB.

    Returns
    -------
    signal_scaled:
        Scaled signal.
    gain:
        Gain used to scale the signal.
    """
    rms_max = rms(signal).max()
    gain = 10**(rms_dB/20)/rms_max
    signal_scaled = gain*signal
    return signal_scaled, gain


def add_decay(brir, rt60, drr, delay, fs, color):
    """
    Decaying noise tail.

    Adds a decaying noise tail to an anechoic BRIR to model room reverberation.
    The output BRIR can then be used to artificially add room reverberation to
    an anechoic signal.

    Parameters
    ----------
    brir : array_like
        Input BRIR. Shape `(len(brir), 2)`.
    rt60 : float
        Reverberation time in seconds.
    drr : float
        Direct-to-reverberant ratio (DRR) in dB.
    delay : float
        Delay in seconds between the highest peak of the input BRIR and the
        start of the decaying noise tail. The peak of the BRIR is calculated
        by taking the earliest of the highest peak in each channel.
    fs : float or int
        Sampling frequency.
    color : {'brown', 'pink', 'white', 'blue', 'violet'}
        Color of the decaying noise tail.

    Returns
    -------
    output_brir :
        BRIR with added decaying noise tail. It is usually longer than `brir`,
        and the exact length depends on `rt60` and `delay`.
    """
    if rt60 == 0:
        return brir
    n = max(int(round(2*(rt60+delay)*fs)), len(brir))
    offset = min(np.argmax(abs(brir), axis=0))
    i_start = int(round(delay*fs)) + offset
    brir_padded = np.zeros((n, 2))
    brir_padded[:len(brir)] = brir
    t = np.arange(n-i_start).reshape(-1, 1)/fs
    noise = colored_noise(color, n-i_start).reshape(-1, 1)
    decaying_tail = np.zeros((n, 2))
    decaying_tail[i_start:] = np.exp(-t/rt60*3*np.log(10))*noise
    decaying_tail, _ = adjust_snr(brir_padded, decaying_tail, drr)
    return brir_padded + decaying_tail


class Mixture:
    """
    Mixture object.

    A convenience class for creating a mixture and accessing components of a
    mixture. The different components (foreground, background, target and
    noise) are calculated on the fly when accessed from the most elementary
    components, namely the early speech, the late speech, the directional
    noise, and the diffuse noise. This allows to reduce the amount of arrays
    in memory.

    E.g. when accessing the `background` attribute, the output is calculated as
    the sum of the late speech, the diretional noise and the diffuse noise.
    """
    def __init__(self):
        self.early_target = None
        self.late_target = None
        self.dir_noise = None
        self.diffuse_noise = None
        self.target_indices = None

    @property
    def mixture(self):
        return self.target + self.noise

    @property
    def target(self):
        return self.early_target + self.late_target

    @property
    def noise(self):
        output = np.zeros(self.shape)
        if self.dir_noise is not None:
            output += self.dir_noise
        if self.diffuse_noise is not None:
            output += self.diffuse_noise
        return output

    @property
    def foreground(self):
        return self.early_target

    @property
    def background(self):
        return self.late_target + self.noise

    @property
    def shape(self):
        return self.early_target.shape

    def __len__(self):
        return len(self.early_target)

    def add_target(self, x, brir, rb, t_pad, fs):
        brir_early, brir_late = split_brir(brir, rb, fs)
        n_pad = round(t_pad*fs)
        self.target_indices = (n_pad, n_pad+len(x))
        x = pad(x, n_pad, where='both')
        self.early_target = spatialize(x, brir_early)
        self.late_target = spatialize(x, brir_late)
        self.early_target = pad(self.early_target, n_pad, where='both')
        self.late_target = pad(self.late_target, n_pad, where='both')

    def add_dir_noises(self, xs, brirs):
        if len(xs) != len(brirs):
            raise ValueError('xs and brirs must have same number of elements')
        self.dir_noise = np.zeros(self.shape)
        for x, brir in zip(xs, brirs):
            self.dir_noise += spatialize(x, brir)

    def add_diffuse_noise(self, brirs, color, ltas=None):
        self.diffuse_noise = np.zeros(self.shape)
        for brir in brirs:
            noise = colored_noise(color, len(self))
            self.diffuse_noise += spatialize(noise, brir)
        if ltas is not None:
            self.diffuse_noise = match_ltas(self.diffuse_noise, ltas)

    def adjust_dir_to_diff_snr(self, snr):
        self.diffuse_noise, _ = adjust_snr(
            self.dir_noise,
            self.diffuse_noise,
            snr,
        )

    def adjust_target_snr(self, snr):
        _, gain = adjust_snr(
            self.target,
            self.noise,
            snr,
            slice(*self.target_indices)
        )
        if self.dir_noise is not None:
            self.dir_noise *= gain
        if self.diffuse_noise is not None:
            self.diffuse_noise *= gain

    def rms(self):
        rms_dB = 20*np.log10(rms(self.mix).max())
        return rms_dB

    def adjust_rms(self, rms_dB):
        _, gain = adjust_rms(self.mix, rms_dB)
        self.early_target *= gain
        self.late_target *= gain
        if self.dir_noise is not None:
            self.dir_noise *= gain
        if self.diffuse_noise is not None:
            self.diffuse_noise *= gain

    def transform(self, transform_func):
        for attr_name in [
                    'early_target',
                    'late_target',
                    'dir_noise',
                    'diffuse_noise',
                ]:
            attr_val = getattr(self, attr_name)
            if attr_val is not None:
                setattr(self, attr_name, transform_func(attr_val))

    def get_long_term_label(self, label='tmr'):
        target = self.early_target
        if label == 'tmr':
            masker = self.late_target + self.noise
        elif label == 'tnr':
            masker = self.noise
        elif label == 'trr':
            masker = self.late_target
        else:
            raise ValueError(f'Label must be tmr, tnr or trr, got {label}')
        slice_ = slice(*self.target_indices)
        energy_target = np.sum(target[slice_].mean(axis=-1)**2)
        energy_masker = np.sum(masker[slice_].mean(axis=-1)**2)
        label = energy_target / (energy_target + energy_masker)
        return label


class RandomPool:
    """
    A pool of elements to randomly draw from. Has it's own random generator.
    Supports weights for non-uniform probability distribution.

    Useful to generate random datasets that differ only along specific
    hyperparameters. For the same seed, we want two datasets to be identical
    along dimensions with common hyperparameter values (e.g. if the set
    of noise types is the same, the random noise recordings should be the
    same). Using the same random generator for the randomization of all
    dimensions would break this.
    """
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
    """
    Like RandomPool, but multiple elements can be drawn simultaneously, with
    replacement. Each extra element is drawn from a dedicated random
    generator. This means that for the same seed, drawing more elements will
    not change the sequence for elements drawn.

    E.g.: Pool is [1, 2, 3]. First experiment draws 2 elements twice and
    obtains [1, 3] and [1, 2]. Second experiment has the same seed but draws 3
    elements twice. Obtains [1, 3, 3] and [1, 2, 1]. Notice the first two
    elements in each draw are the same. If we were using the same random
    generator for every single number, the third number in the first draw
    would have changed the seed, and the second draw would have thus been
    completely different.
    """
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


class ContinuousRandomGenerator:
    """
    Like RandomPool, but for continuous distributions.
    """
    def __init__(self, dist_name, dist_args, seed=None):
        self.random = np.random.RandomState(seed)
        self.dist_name = dist_name
        self.dist_args = dist_args

    def get(self):
        dist_func = getattr(self.random, self.dist_name)
        return dist_func(*self.dist_args)


class Seeder:
    """
    An integer number generator. Used to generate seeds for other random
    generators.
    """
    def __init__(self, seed_value, max_seed=100):
        self.random = random.Random(seed_value)
        self.max_seed = max_seed

    def get(self):
        return self.random.randrange(self.max_seed)


class RandomMixtureMaker:
    def __init__(
                self,
                fs,
                rooms,
                speakers,
                target_snr_dist_name,
                target_snr_dist_args,
                target_angle_min,
                target_angle_max,
                dir_noise_nums,
                dir_noise_types,
                dir_noise_snrs,
                dir_noise_angle_min,
                dir_noise_angle_max,
                diffuse_noise_on,
                diffuse_noise_color,
                diffuse_noise_ltas_eq,
                mixture_pad,
                mixture_rb,
                mixture_rms_jitter_on,
                mixture_rms_jitters,
                filelims_target,
                filelims_dir_noise,
                filelims_room,
                decay_on,
                decay_color,
                decay_rt60s,
                decay_drr_dist_name,
                decay_drr_dist_args,
                decay_delays,
                seed_on,
                seed_value,
                uniform_tmr,
            ):

        if not seed_on:
            seed_value = None
        seeder = Seeder(seed_value)

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
        # LIBRI material.
        if len(speakers) > 1:
            logging.info('Calculating each corpus average duration to weight '
                         'probabilities')
            weights = {speaker: 1/get_average_duration(speaker, self.def_cfg)
                       for speaker in speakers}
        else:
            weights = {speaker: 1 for speaker in speakers}
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
        self.mix = Mixture()
        self.metadata = {}
        self.room = self.get_random_room()
        self.decayer = self.get_random_decayer()
        self.add_random_target()
        self.add_random_dir_noises()
        self.add_random_diffuse_noise()
        self.set_random_dir_to_diff_snr()
        self.set_random_target_snr()
        self.set_random_tmr()
        self.set_random_rms()
        self.get_long_term_labels()
        return self.mix, self.metadata

    def get_random_room(self):
        room_regexp = self.room_regexps.get()
        self.rooms.set_pool(get_rooms(room_regexp))
        room = self.rooms.get()
        self.metadata['room'] = room
        return room

    def get_random_decayer(self):
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
            self.metadata['decay'] = {}
            self.metadata['decay']['rt60'] = rt60
            self.metadata['decay']['drr'] = drr
            self.metadata['decay']['delay'] = delay
            self.metadata['decay']['color'] = self.decay_color
        return decayer

    def add_random_target(self):
        angles = get_available_angles(
            self.room,
            def_cfg=self.def_cfg,
            angle_min=self.target_angle_min,
            angle_max=self.target_angle_max,
            parity=self.filelims_room,
        )
        self.target_angles.set_pool(angles)
        angle = self.target_angles.get()
        brir = self._load_brirs(self.room, angle)
        brir = self.decayer.run(brir)
        speaker = self.speakers.get()
        target, filename = load_random_target(
            speaker,
            self.filelims_target,
            self.fs,
            randomizer=self.target_filename_randomizer.random,
            def_cfg=self.def_cfg,
            all_filepaths=self.all_filepaths_dict[speaker]
        )
        self.mix.add_target(
            x=target,
            brir=brir,
            rb=self.mixture_rb,
            t_pad=self.mixture_pad,
            fs=self.fs,
        )
        self.metadata['target'] = {}
        self.metadata['target']['angle'] = angle
        self.metadata['target']['filename'] = filename

    def add_random_dir_noises(self):
        number = self.dir_noise_nums.get()
        types = self.dir_noise_types.get(number)

        if types and types[0] == 'bbl':

            angles = get_available_angles(self.room, self.def_cfg)
            if len(angles) == 1:
                raise ValueError('cannot use bbl noise with a room that only '
                                 'has one brir')
            angles = [a for a in angles
                      if a != self.metadata['target']['angle']]
            brirs = self._load_brirs(self.room, angles)
            brirs = [self.decayer.run(brir) for brir in brirs]
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
                while len(noise) < len(self.mix):
                    noise_, _ = load_random_target(
                        speaker,
                        self.filelims_target,
                        self.fs,
                        randomizer=self.bbl_filename_randomizer.random,
                        def_cfg=self.def_cfg,
                        all_filepaths=self.all_filepaths_dict[speaker]
                    )
                    noise = np.hstack((noise, noise_))
                noise = noise[:len(self.mix)]
                noises.append(noise)
            self.mix.add_dir_noises(noises, brirs)
            self.metadata['directional'] = {}
            self.metadata['directional']['number'] = len(angles)
            self.metadata['directional']['sources'] = []

        else:

            types = [t for t in types if t != 'bbl']
            number = len(types)
            angles = get_available_angles(
                self.room,
                def_cfg=self.def_cfg,
                angle_min=self.dir_noise_angle_min,
                angle_max=self.dir_noise_angle_max,
                parity=self.filelims_room,
            )
            if len(angles) > 1:
                angles = [a for a in angles if a !=
                          self.metadata['target']['angle']]
            self.dir_noise_angles.set_pool(angles)
            angles = self.dir_noise_angles.get(number)
            noises, files, indices = self._load_noises(
                types,
                len(self.mix),
            )
            brirs = self._load_brirs(self.room, angles)
            brirs = [self.decayer.run(brir) for brir in brirs]
            self.mix.add_dir_noises(noises, brirs)
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

    def add_random_diffuse_noise(self):
        if self.diffuse_noise_on:
            brirs = self._load_brirs(self.room)
            self.mix.add_diffuse_noise(
                brirs,
                self.diffuse_noise_color,
                self.ltas,
            )
            self.metadata['diffuse'] = {}
            self.metadata['diffuse']['color'] = self.diffuse_noise_color
            self.metadata['diffuse']['ltas_eq'] = self.diffuse_noise_ltas_eq

    def set_random_dir_to_diff_snr(self):
        snr = self.dir_noise_snrs.get()
        if self.metadata['directional']['number'] == 0:
            return
        if self.diffuse_noise_on:
            self.mix.adjust_dir_to_diff_snr(snr)
            self.metadata['directional']['snr'] = snr

    def set_random_target_snr(self):
        snr = self.target_snrs.get()
        if self.metadata['directional']['number'] == 0:
            if not self.diffuse_noise_on:
                return
        self.mix.adjust_target_snr(snr)
        self.metadata['target']['snr'] = snr

    def set_random_tmr(self):
        tmr = self.tmrs.get()
        alpha = self.tmrs.get()
        if self.uniform_tmr:
            target_energy = np.sum(self.mix.early_target.mean(axis=1)**2)
            new_masker_energy = target_energy*(1/tmr-1)
            new_noise_energy = alpha*new_masker_energy
            new_reverb_energy = (1-alpha)*new_masker_energy
            cur_noise_energy = np.sum(self.mix.noise.mean(axis=1)**2)
            cur_reverb_energy = np.sum(self.mix.late_target.mean(axis=1)**2)
            noise_gain = (new_noise_energy/cur_noise_energy)**0.5
            reverb_gain = (new_reverb_energy/cur_reverb_energy)**0.5
            if self.mix.dir_noise is not None:
                self.mix.dir_noise = noise_gain*self.mix.dir_noise
            if self.mix.diffuse_noise is not None:
                self.mix.diffuse_noise = noise_gain*self.mix.diffuse_noise
            self.mix.late_target = reverb_gain*self.mix.late_target
            self.metadata['uniform_tmr'] = {}
            self.metadata['uniform_tmr']['tmr'] = tmr
            self.metadata['uniform_tmr']['alpha'] = alpha
            self.metadata['target']['snr'] = None
            self.metadata['decay']['drr'] = None

    def set_random_rms(self):
        rms_dB = self.mixture_rms_jitters.get()
        if self.mixture_rms_jitter_on:
            rms_start = self.mix.rms()
            self.mix.adjust_rms(rms_start + rms_dB)
            self.metadata['rms_dB'] = rms_dB

    def get_long_term_labels(self):
        self.metadata['lt_labels'] = {}
        for label in ['tmr', 'tnr', 'trr']:
            long_term_label = self.mix.get_long_term_label(label)
            self.metadata['lt_labels'][label] = long_term_label

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
    """
    A convenience class that calls add_decay with attributes as arguments
    """
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
