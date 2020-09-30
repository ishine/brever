import numpy as np
import scipy.signal

from .utils import zero_pad, fft_freqs, rms


def spatialize(x, brir):
    '''
    Spatialize an input audio signal.

    Parameters:
        x:
            Monaural audio signal to spatialize.
        brir:
            Binaural room impulse response.

    Returns:
        x_binaural:
            Binaural audio signal. Shape len(x)*2.
    '''
    n = len(x)
    x_left = scipy.signal.oaconvolve(x, brir[:, 0], mode='full')[:n]
    x_right = scipy.signal.oaconvolve(x, brir[:, 1], mode='full')[:n]
    return np.vstack([x_left, x_right]).T


def colored_noise(color, n_samples):
    '''
    Generate 1/f**alpha colored noise.

    Parameters:
        color:
            Color of the noise. Can be 'brown', 'pink', 'white', 'blue' or
            'violet'.
        n_samples:
            Number of samples to generate.

    Returns:
        x:
            Colored noise. Length n_samples.
    '''
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
    '''
    Splits a BRIR into a direct or early reflections component and a reverb or
    late reflections component.

    Parameters:
        brir:
            Input BRIR.
        reflection_boundary:
            Reflection boundary defining the limit between early and late
            reflections.
        fs:
            Sampling frequency.
        max_itd:
            Maximum interaural time difference. Used to compare the locations
            of the peak in each channel and correct if necessary.

    Returns:
        brir_early:
            Early reflections part of input BRIR. Same length as brir.
        brir_late:
            Late reflections part of input BRIR. Same length as brir.
    '''
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
    '''
    Scales a noise signal given a target signal and a desired SNR.

    Parameters:
        signal:
            Target signal.
        noise:
            Noise signal.
        snr:
            Desired SNR.
        slice_:
            Slice of the input signal from which the SNR should be calculated.
            If left as None, the energy of the entire signals are calculated.

    Returns:
        noise_scaled:
            Scaled noise. The SNR between the target signal and the new scaled
            noise equals snr.
    '''
    if slice_ is None:
        slice_ = np.s_[:]
    energy_signal = np.sum(signal[slice_].mean(axis=1)**2)
    energy_noise = np.sum(noise[slice_].mean(axis=1)**2)
    if energy_signal == 0:
        raise ValueError('Can\'t scale noise signal if target signal is 0!')
    if energy_noise == 0:
        raise ValueError('Can\'t scale noise signal if it equals 0!')
    gain = (10**(-snr/10)*energy_signal/energy_noise)**0.5
    noise_scaled = gain*noise
    return noise_scaled, gain


def adjust_rms(signal, rms_dB):
    '''
    Scales a signal such that it presents a given RMS in dB.

    Parameters:
        signal:
            Input signal.
        rms_dB:
            Desired RMS in dB.

    Returns:
        signal_scaled:
            Scaled signal.
        gain:
            Gain used to scale the signal.
    '''
    rms_max = rms(signal).max()
    gain = 10**(rms_dB/20)/rms_max
    signal_scaled = gain*signal
    return signal_scaled, gain


def add_decay(brir, rt60, drr, delay, fs, color):
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

    def add_target(self, x, brir, rb, pad, fs):
        brir_early, brir_late = split_brir(brir, rb, fs)
        n_pad = round(pad*fs)
        self.target_indices = (n_pad, n_pad+len(x))
        x = zero_pad(x, n_pad, 'both')
        self.early_target = spatialize(x, brir_early)
        self.late_target = spatialize(x, brir_late)
        self.early_target = zero_pad(self.early_target, n_pad, 'both')
        self.late_target = zero_pad(self.late_target, n_pad, 'both')

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
