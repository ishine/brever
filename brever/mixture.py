import random
import re

import numpy as np
import scipy.signal

from .io import AudioFileLoader
from .utils import fft_freqs, pad


def rms(x, axis=0):
    """
    Root mean square.

    Calculates the root mean square (RMS) of input array along given axis.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which to calculate. Default is 0.

    Returns
    -------
    rms : array_like
        RMS values.
    """
    return np.mean(x**2, axis=axis)**0.5


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
        raise ValueError('cannot scale noise signal if target signal is 0')
    if energy_noise == 0:
        raise ValueError('cannot scale noise signal if it equals 0')
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
        self.diffuse = None
        self.target_idx = None

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
        if self.diffuse is not None:
            output += self.diffuse
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

    def add_target(self, x, brir, reflection_boundary, padding, fs):
        brir_early, brir_late = split_brir(brir, reflection_boundary, fs)
        n_pad = round(padding*fs)
        self.target_idx = (n_pad, n_pad+len(x))
        x = pad(x, n_pad, where='both')
        self.early_target = spatialize(x, brir_early)
        self.late_target = spatialize(x, brir_late)
        self.early_target = pad(self.early_target, n_pad, where='both')
        self.late_target = pad(self.late_target, n_pad, where='both')

    def add_noises(self, xs, brirs):
        if len(xs) != len(brirs):
            raise ValueError('xs and brirs must have same number of elements')
        if not xs:
            raise ValueError('xs and brirs cannot be empty')
        self.dir_noise = np.zeros(self.shape)
        for x, brir in zip(xs, brirs):
            self.dir_noise += spatialize(x, brir)

    def add_diffuse_noise(self, brirs, color, ltas=None):
        if not brirs:
            raise ValueError('brirs cannot be empty')
        self.diffuse = np.zeros(self.shape)
        for brir in brirs:
            noise = colored_noise(color, len(self))
            self.diffuse += spatialize(noise, brir)
        if ltas is not None:
            self.diffuse = match_ltas(self.diffuse, ltas)

    def set_ndr(self, ndr):
        self.diffuse, _ = adjust_snr(
            self.dir_noise,
            self.diffuse,
            ndr,
        )

    def set_snr(self, snr):
        _, gain = adjust_snr(
            self.target,
            self.noise,
            snr,
            slice(*self.target_idx)
        )
        if self.dir_noise is not None:
            self.dir_noise *= gain
        if self.diffuse is not None:
            self.diffuse *= gain

    def set_rms(self, rms_dB):
        _, gain = adjust_rms(self.mixture, rms_dB)
        self.early_target *= gain
        self.late_target *= gain
        if self.dir_noise is not None:
            self.dir_noise *= gain
        if self.diffuse is not None:
            self.diffuse *= gain

    def get_rms(self):
        rms_dB = 20*np.log10(rms(self.mixture).max())
        return rms_dB

    def transform(self, func):
        for attr_name in [
                    'early_target',
                    'late_target',
                    'noise',
                    'diffuse',
                ]:
            attr_val = getattr(self, attr_name)
            if attr_val is not None:
                setattr(self, attr_name, func(attr_val))

    def get_long_term_label(self, label='tmr'):
        target = self.early_target
        if label == 'tmr':
            masker = self.late_target + self.noise
        elif label == 'tnr':
            masker = self.noise
        elif label == 'trr':
            masker = self.late_target
        else:
            raise ValueError(f'label must be tmr, tnr or trr, got {label}')
        slice_ = slice(*self.target_idx)
        energy_target = np.sum(target[slice_].mean(axis=-1)**2)
        energy_masker = np.sum(masker[slice_].mean(axis=-1)**2)
        label = energy_target / (energy_target + energy_masker)
        return label

    def scale_background(self, gain):
        self.late_target = gain*self.late_target
        if self.dir_noise is not None:
            self.dir_noise = gain*self.dir_noise
        if self.diffuse is not None:
            self.diffuse = gain*self.diffuse


class BaseRandGen:
    """
    Base class for all random generators. The __init__() and roll() methods
    should be overwritten to obtain desired behaviors.

    The subsequent classes are useful to generate random datasets that differ
    only along specific hyperparameters. For the same seed, we want two
    datasets to be identical along dimensions with common hyperparameter
    values (e.g. if the set of noise types is the same, the random noise
    recordings should be the same). Using the same random generator for the
    randomization of all dimensions would break this.
    """
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed)
        self._to_yield = None

    def roll(self):
        self._to_yield = self.random.rand()

    def get(self):
        if self._to_yield is None:
            raise ValueError('must call roll() before calling get()')
        output = self._to_yield
        self._to_yield = None
        return output


class ChoiceRandGen(BaseRandGen):
    """
    A pool of elements to randomly draw from. Supports weights for non-uniform
    probability distribution. Supports multiple draws, with or without
    replacement.

    When drawing more than one element, each extra element is drawn from a
    dedicated random generator. This means that for the same seed, drawing
    more elements will not change the sequence of elements drawn. This is why
    we can't use np.random.choice with size > 1 .

    E.g.: Pool is [1, 2, 3]. First experiment draws 2 elements twice and
    obtains [1, 3] and [1, 2]. Second experiment has the same seed but draws 3
    elements twice. Obtains [1, 3, 3] and [1, 2, 1]. Notice the first two
    elements in each draw are the same. If we were using the same random
    generator for every single number, the third number in the first draw
    would have changed the seed, and the second draw would have thus been
    completely different.
    """
    def __init__(self, pool, size=1, weights=None, replace=True, seed=None):
        super().__init__(seed)
        self.random = [
            np.random.RandomState(
                seed if seed is None else seed+i
            )
            for i in range(size)
        ]
        if isinstance(pool, set):
            self.pool = sorted(pool)
            if weights is not None:
                if not isinstance(weights, dict):
                    raise ValueError('weights must be dict when pool is set')
                if set(weights.keys()) != pool:
                    raise ValueError('weights keys do not match pool')
                weights = [weights[x] for x in self.pool]
        else:
            self.pool = pool
            if weights is not None:
                if not isinstance(weights, list):
                    raise ValueError('weights must be list when pool is list')
                if len(weights) != len(pool):
                    raise ValueError('weights and pool must have same length')
        if weights is not None:
            weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.replace = replace

    def roll(self):
        self._to_yield = []
        current_pool = self.pool.copy()
        for rand in self.random:
            val = rand.choice(current_pool, p=self.weights).item()
            self._to_yield.append(val)
            if not self.replace:
                current_pool.remove(val)
        if len(self._to_yield) == 1:
            self._to_yield, = self._to_yield


class DistRandGen(BaseRandGen):
    """
    For arbitrary distributions as provided by np.random.
    """
    def __init__(self, dist_name, dist_args, seed=None):
        super().__init__(seed)
        self.dist_name = dist_name
        self.dist_args = dist_args

    def roll(self):
        dist_func = getattr(self.random, self.dist_name)
        self._to_yield = dist_func(*self.dist_args)


class MultiChoiceRandGen(BaseRandGen):
    """
    Interface for multiple ChoiceRandGen objects that are accessed via keys.
    """
    def __init__(self, pool_dict, size=1, replace=True, seed=None):
        self.random = {}
        if not pool_dict:
            raise ValueError('pool_dict cannot be empty')
        for i, key in enumerate(sorted(pool_dict.keys())):
            self.random[key] = ChoiceRandGen(
                pool=pool_dict[key],
                size=size,
                replace=replace,
                seed=seed if seed is None else seed+i,
            )
        self._to_yield = None

    def roll(self):
        self._to_yield = {}
        for key, rand in self.random.items():
            rand.roll()
            self._to_yield[key] = rand.get()

    def get(self, key):
        if self._to_yield is None:
            raise ValueError('must call roll() before calling get()')
        if isinstance(key, list):
            list_input = True
        else:
            key = [key]
            list_input = False
        output = [self._to_yield[key] for key in key]
        self._to_yield = None
        if not list_input:
            output, = output
        return output


class AngleRandomizer:
    """
    Interface for target and noise angle randomizers.

    Note: the target can be collocated with one noise source. This is because
    using different angle limits for the target and the noise creates different
    angle pools for the target and the noise. It's impossible to design two
    independent randomizers with different pools that always generate
    different items.
    """
    def __init__(self, pool_dict, target_angle=None, noise_angle=None,
                 noise_num=1, parity='all', seed=None):
        if not pool_dict:
            raise ValueError('pool_dict cannot be empty')
        target_pool_dict = {
            room: self.filter_angles(
                angles, target_angle, parity,
            )
            for room, angles in pool_dict.items()
        }
        noise_pool_dict = {
            room: self.filter_angles(
                angles, noise_angle, parity,
            ) for room, angles in pool_dict.items()
        }
        self.target_random = MultiChoiceRandGen(
            pool_dict=target_pool_dict,
            seed=seed,
        )
        self.noise_random = MultiChoiceRandGen(
            pool_dict=noise_pool_dict,
            size=noise_num,
            replace=False,
            seed=seed if seed is None else seed+1,
        )

    def roll(self):
        self.target_random.roll()
        self.noise_random.roll()

    def get(self, key):
        return self.target_random.get(key), self.noise_random.get(key)

    def filter_angles(self, angles, angle_lims, parity):
        angles = sorted(angles)
        if parity == 'all':
            pass
        elif parity == 'even' or parity == 'odd':
            even_angles = angles[::2]
            odd_angles = angles[1::2]
            if 0 not in even_angles:
                even_angles, odd_angles = odd_angles, even_angles
            if parity == 'even':
                angles = even_angles
            else:
                angles = odd_angles
        else:
            raise ValueError(f'parity must be all, odd or even, got {parity}')
        if angle_lims is not None:
            angle_min, angle_max = angle_lims
            angles = [a for a in angles if angle_min <= a <= angle_max]
        return angles


class TargetFileRandomizer(MultiChoiceRandGen):
    """
    Interface to randomize target files. Basically a MultiChoiceRandGen object
    with file limit support.
    """
    def __init__(self, pool_dict, *args, lims=[0.0, 1.0], **kwargs):
        super().__init__(
            pool_dict=self.make_pool_dict(pool_dict, lims),
            *args,
            **kwargs,
        )

    def make_pool_dict(self, pool_dict, lims):
        output = {}
        for key, files in pool_dict.items():
            n = len(files)
            i_min, i_max = round(n*lims[0]), round(n*lims[1])
            output[key] = files[i_min:i_max]
        return output


class NoiseFileRandomizer(MultiChoiceRandGen):
    """
    Interface to randomize noise files. Basically a MultiChoiceRandGen object
    with file limit support and a sensibly different get() method. The get()
    method can be called as many times as the roll size.
    """
    def __init__(self, pool_dict, *args, lims=[0.0, 1.0], size=1, **kwargs):
        super().__init__(
            pool_dict=self.make_pool_dict(pool_dict, lims),
            *args,
            size=size,
            **kwargs,
        )
        self.size = size
        self.counter = [False]*self.size

    def make_pool_dict(self, pool_dict, lims):
        output = {}
        for key, files in pool_dict.items():
            if not is_long_recording(key):
                n = len(files)
                i_min, i_max = round(n*lims[0]), round(n*lims[1])
                files = files[i_min:i_max]
            output[key] = files
        return output

    def roll(self):
        super().roll()
        self.counter = [False]*self.size

    def get(self, noise, idx):
        if self._to_yield is None or self.counter[idx]:
            raise ValueError('must call roll() before calling get()')
        output = self._to_yield[noise][idx]
        self.counter[idx] = True
        if all(self.counter):
            self._to_yield = None
            self.counter = [False]*self.size
        return output


class Seeder:
    """
    An integer number generator. Used to generate seeds for other random
    generators.
    """
    def __init__(self, seed, max_seed=100):
        self.random = random.Random(seed)
        self.max_seed = max_seed

    def get(self):
        return self.random.randrange(self.max_seed)


class RandomMixtureMaker:
    """
    Main mixture maker object.
    """
    def __init__(
        self,
        fs=16e3,
        seed=None,
        padding=0.0,
        uniform_tmr=False,
        reflection_boundary=0.05,
        speakers={'timit_.*'},
        noises={'dcase_.*'},
        rooms={'surrey_.*'},
        snr_dist_name='uniform',
        snr_dist_args=[-5.0, 10.0],
        target_angle=[-90.0, 90.0],
        noise_num=[1, 3],
        noise_angle=[-90.0, 90.0],
        ndr_dist_name='uniform',
        ndr_dist_args=[0.0, 30.0],
        diffuse=False,
        diffuse_color='white',
        diffuse_ltas_eq=False,
        decay=False,
        decay_color='white',
        rt60_dist_name='uniform',
        rt60_dist_args=[0.1, 5.0],
        drr_dist_name='uniform',
        drr_dist_args=[5.0, 35.0],
        delay_dist_name='uniform',
        delay_dist_args=[0.075, 0.100],
        rms_jitter_dist_name='uniform',
        rms_jitter_dist_args=[0.0, 0.0],
        speech_files=[0.0, 1.0],
        noise_files=[0.0, 1.0],
        room_files='all',
    ):

        seeder = Seeder(seed)

        # init audio file loader
        self.loader = AudioFileLoader(fs)
        self.loader.scan_material(speakers, noises, rooms)

        self.fs = fs
        self.padding = padding
        self.reflection_boundary = reflection_boundary
        self.speech_files = speech_files
        self.noise_files = noise_files
        self.room_files = 'all'

        # speakers
        """
        For the speakers random generator, the probability distribution must be
        weighted according to the average duration of the sentences, otherwise
        the speech material in the dataset will be unbalanced. Example: TIMIT
        sentences are 3 seconds long on average, while LIBRI sentences are 12
        seconds long on average, so making a dataset using 50 TIMIT sentences
        and 50 LIBRI sentences will result in much more LIBRI material.
        """
        weights = self.loader.calc_weights(speakers)
        self.speakers = ChoiceRandGen(
            pool=speakers,
            weights=weights,
            seed=seeder.get(),
        )

        # noises
        self.noises = ChoiceRandGen(
            pool=noises,
            size=noise_num[1],
            seed=seeder.get(),
        )

        # rooms
        self.room_regexps = ChoiceRandGen(
            pool=rooms,
            seed=seeder.get(),
        )
        self.rooms = MultiChoiceRandGen(
            pool_dict=self.loader._room_regexps,
            seed=seeder.get(),
        )

        # angles
        self.angles = AngleRandomizer(
            pool_dict=self.loader._room_angles,
            target_angle=target_angle,
            noise_angle=noise_angle,
            noise_num=noise_num[1],
            parity=room_files,
            seed=seeder.get(),
        )

        # target parameters
        self.snrs = DistRandGen(
            dist_name=snr_dist_name,
            dist_args=snr_dist_args,
            seed=seeder.get()
        )
        self.target_file_randomizer = TargetFileRandomizer(
            pool_dict=self.loader._speech_files,
            lims=speech_files,
            seed=seeder.get(),
        )

        # noise parameters
        self.noise_ndrs = DistRandGen(
            dist_name=ndr_dist_name,
            dist_args=ndr_dist_args,
            seed=seeder.get()
        )
        self.noise_nums = DistRandGen(
            dist_name='randint',
            dist_args=[noise_num[0], noise_num[1]+1],
            seed=seeder.get(),
        )
        self.noise_file_randomizer = NoiseFileRandomizer(
            pool_dict=self.loader._noise_files,
            lims=noise_files,
            size=noise_num[1],
            replace=False,
            seed=seeder.get(),
        )

        # diffuse noise parameters
        self.diffuse = diffuse
        self.diffuse_color = diffuse_color
        self.diffuse_ltas_eq = diffuse_ltas_eq

        # decay parameters
        self.decay = decay
        self.decay_color = decay_color
        self.decay_rt60s = DistRandGen(
            dist_name=rt60_dist_name,
            dist_args=rt60_dist_args,
            seed=seeder.get(),
        )
        self.decay_drrs = DistRandGen(
            dist_name=drr_dist_name,
            dist_args=drr_dist_args,
            seed=seeder.get(),
        )
        self.decay_delays = DistRandGen(
            dist_name=delay_dist_name,
            dist_args=delay_dist_args,
            seed=seeder.get(),
        )

        # rms jitter parameters
        self.rms_jitters = DistRandGen(
            dist_name=rms_jitter_dist_name,
            dist_args=rms_jitter_dist_args,
            seed=seeder.get()
        )

        # tmr randomizers
        self.uniform_tmr = uniform_tmr
        self.tmrs = DistRandGen(
            dist_name='uniform',
            dist_args=[0.0, 1.0],
            seed=seeder.get(),
        )

        # calculate ltas now for efficiency
        if ((diffuse and diffuse_ltas_eq)
                or ('ssn' in noises and noise_num[1] > 0)):
            self.ltas = self.loader.calc_ltas(speakers)
        else:
            self.ltas = None

    def make(self):
        self.roll()
        self.mix = Mixture()
        self.metadata = {}
        self.room = self.get_random_room()
        self.decayer = self.get_random_decayer()
        target_angle, noise_angles = self.angles.get(self.room)
        self.add_random_target(target_angle)
        self.add_random_noises(noise_angles)
        self.add_random_diffuse_noise()
        self.set_random_ndr()
        self.set_random_snr()
        self.set_random_tmr()
        self.set_random_rms()
        self.calc_long_term_labels()
        return self.mix, self.metadata

    def roll(self):
        self.speakers.roll()
        self.noises.roll()
        self.room_regexps.roll()
        self.rooms.roll()
        self.angles.roll()
        self.snrs.roll()
        self.target_file_randomizer.roll()
        self.noise_ndrs.roll()
        self.noise_nums.roll()
        self.noise_file_randomizer.roll()
        self.decay_rt60s.roll()
        self.decay_drrs.roll()
        self.decay_delays.roll()
        self.rms_jitters.roll()
        self.tmrs.roll()

    def get_random_room(self):
        room_regexp = self.room_regexps.get()
        room = self.rooms.get(room_regexp)
        self.metadata['room'] = room
        return room

    def get_random_decayer(self):
        rt60 = self.decay_rt60s.get()
        drr = self.decay_drrs.get()
        delay = self.decay_delays.get()
        decayer = Decayer(
            rt60=rt60,
            drr=drr,
            delay=delay,
            fs=self.fs,
            color=self.decay_color,
            active=self.decay,
        )
        if self.decay:
            self.metadata['decay'] = {}
            self.metadata['decay']['rt60'] = rt60
            self.metadata['decay']['drr'] = drr
            self.metadata['decay']['delay'] = delay
            self.metadata['decay']['color'] = self.decay_color
        return decayer

    def add_random_target(self, angle):
        speaker = self.speakers.get()
        file = self.target_file_randomizer.get(speaker)
        x = self.loader.load_file(file)
        brir, _ = self.loader.load_brirs(self.room, angle)
        brir = self.decayer.run(brir)
        self.mix.add_target(
            x=x,
            brir=brir,
            reflection_boundary=self.reflection_boundary,
            padding=self.padding,
            fs=self.fs,
        )
        self.metadata['target'] = {}
        self.metadata['target']['angle'] = angle
        self.metadata['target']['file'] = file

    def add_random_noises(self, angles):
        number = self.noise_nums.get()
        angles = angles[:number]
        noises = self.noises.get()[:number]
        xs, files, idxs = self.make_noises(noises)
        brirs, _ = self.loader.load_brirs(self.room, angles)
        brirs = [self.decayer.run(brir) for brir in brirs]
        self.mix.add_noises(xs, brirs)
        self.metadata['noises'] = []
        for i in range(number):
            self.metadata['noises'].append({
                'angle': angles[i],
                'type': noises[i],
                'file': files[i],
                'idx': idxs[i],
            })

    def add_random_diffuse_noise(self):
        if self.diffuse:
            brirs, _ = self.loader.load_brirs(self.room)
            self.mix.add_diffuse_noise(
                brirs=brirs,
                color=self.diffuse_color,
                ltas=self.ltas,
            )
            self.metadata['diffuse'] = {}
            self.metadata['diffuse']['color'] = self.diffuse_color
            self.metadata['diffuse']['ltas_eq'] = self.diffuse_ltas_eq

    def set_random_ndr(self):
        ndr = self.noise_ndrs.get()
        if self.diffuse and self.mix.dir_noise is not None:
            self.mix.set_ndr(ndr)
            self.metadata['noise']['ndr'] = ndr

    def set_random_snr(self):
        snr = self.snrs.get()
        if self.diffuse or self.mix.dir_noise is not None:
            self.mix.set_snr(snr)
            self.metadata['target']['snr'] = snr

    def set_random_tmr(self):
        tmr = self.tmrs.get()
        if self.uniform_tmr:
            target_energy = np.sum(self.mix.foreground.mean(axis=1)**2)
            new_masker_energy = target_energy*(1/tmr-1)
            old_masker_energy = np.sum(self.mix.background.mean(axis=1)**2)
            gain = (new_masker_energy/old_masker_energy)**0.5
            self.mix.scale_background(gain)
            self.metadata['tmr'] = tmr
            self.metadata['target']['snr'] = None
            self.metadata['decay']['drr'] = None

    def set_random_rms(self):
        rms_jitter = self.rms_jitters.get()
        rms = self.mix.get_rms()
        self.mix.set_rms(rms + rms_jitter)
        self.metadata['rms_jitter'] = rms_jitter

    def calc_long_term_labels(self):
        self.metadata['long_term_labels'] = {}
        for label in ['tmr', 'tnr', 'trr']:
            long_term_label = self.mix.get_long_term_label(label)
            self.metadata['long_term_labels'][label] = long_term_label

    def make_noises(self, noises):
        to_zip = []
        for i, noise in enumerate(noises):
            if noise.startswith('colored_'):
                color = re.match('^colored_(.*)$', noise).group(1)
                x = colored_noise(color, len(self.mix))
                file = None
                idx = None
            elif noise == 'ssn':
                x = colored_noise('white', len(self.mix))
                x = match_ltas(x, self.ltas)
                file = None
                idx = None
            else:
                file = self.noise_file_randomizer.get(noise, i)
                lims = self.noise_files if is_long_recording(noise) else None
                x, idx = self.loader.load_noise(
                    file=file,
                    n_samples=len(self.mix),
                    lims=lims,
                )
            to_zip.append((x, file, idx))
        xs, files, idxs = zip(*to_zip)
        return xs, files, idxs


class Decayer:
    """
    Interface for adding decaying tail to BRIRs.
    """
    def __init__(self, rt60, drr, delay, fs, color, active=True):
        self.rt60 = rt60
        self.drr = drr
        self.delay = delay
        self.fs = fs
        self.color = color
        self.active = active

    def run(self, brir):
        if not self.active:
            return brir
        if self.rt60 == 0:
            return brir
        n = max(int(round(2*(self.rt60+self.delay)*self.fs)), len(brir))
        offset = min(np.argmax(abs(brir), axis=0))
        i_start = int(round(self.delay*self.fs)) + offset
        brir_padded = np.zeros((n, 2))
        brir_padded[:len(brir)] = brir
        t = np.arange(n-i_start).reshape(-1, 1)/self.fs
        noise = colored_noise(self.color, n-i_start).reshape(-1, 1)
        decaying_tail = np.zeros((n, 2))
        decaying_tail[i_start:] = np.exp(-t/self.rt60*3*np.log(10))*noise
        decaying_tail, _ = adjust_snr(brir_padded, decaying_tail, self.drr)
        return brir_padded + decaying_tail


def is_long_recording(alias):
    """
    For some databases, the train/test split must be done on the file level
    rather than on the folder level.
    """
    if alias.startswith((
        'noisex',
        'icra',
        'demand',
        'arte',
    )):
        return True
    elif alias.startswith((
        'dcase',
    )):
        return False
    else:
        raise ValueError(f'wrong noise alias, got {alias}')
