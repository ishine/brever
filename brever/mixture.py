import numpy as np
import scipy.signal

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
    n = len(x)
    noverlap = n_fft-hop_length
    _, _, X = scipy.signal.stft(x, nperseg=n_fft, noverlap=noverlap, axis=0)
    ltas_X = np.mean(np.abs(X**2), axis=(1, 2))
    EQ = (ltas/ltas_X)**0.5
    X *= EQ[:, np.newaxis, np.newaxis]
    _, x = scipy.signal.istft(X, nperseg=n_fft, noverlap=noverlap, freq_axis=0)
    x = x.T
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
        self.directional_noise = None
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
        if self.directional_noise is not None:
            output += self.directional_noise
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

    def add_directional_noises(self, xs, brirs):
        if len(xs) != len(brirs):
            raise ValueError('xs and brirs must have same number of elements')
        self.directional_noise = np.zeros(self.shape)
        for x, brir in zip(xs, brirs):
            self.directional_noise += spatialize(x, brir)

    def add_diffuse_noise(self, brirs, color, ltas_eq):
        self.diffuse_noise = np.zeros(self.shape)
        for brir in brirs:
            noise = colored_noise(color, len(self))
            self.diffuse_noise += spatialize(noise, brir)
        if ltas_eq:
            ltas = np.load('ltas.npy')
            self.diffuse_noise = match_ltas(self.diffuse_noise, ltas)

    def adjust_dir_to_diff_snr(self, snr):
        self.diffuse_noise, _ = adjust_snr(
            self.directional_noise,
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
        if self.directional_noise is not None:
            self.directional_noise *= gain
        if self.diffuse_noise is not None:
            self.diffuse_noise *= gain

    def adjust_rms(self, rms_dB):
        _, gain = adjust_rms(self.mixture, rms_dB)
        self.early_target *= gain
        self.late_target *= gain
        if self.directional_noise is not None:
            self.directional_noise *= gain
        if self.diffuse_noise is not None:
            self.diffuse_noise *= gain
