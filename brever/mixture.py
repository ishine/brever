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
    energy_signal = np.sum(signal[slice_]**2)
    energy_noise = np.sum(noise[slice_]**2)
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


def diffuse_and_directional_noise(xs_sources, brirs_sources, xs_diffuse,
                                  brirs_diffuse, snr, ltas):
    '''
    Creates a mixture consisting of a set of directional noise sources in
    diffuse noise at given SNR.

    Parameters:
        x_sources:
            List of clean directional noise signals to convolve with
            brirs_sources. Must all have same length.
        brirs_sources:
            List of BRIRs to convolve each directional noise signal with.
            Must have same length as x_sources.
        xs_diffuse:
            Clean noise signals used to create diffuse noise using
            brirs_diffuse. All the elements must have same length as the
            elements of x_sources.
        brirs_diffuse:
            List of BRIRs to use to create the diffuse noise. Ideally the BRIRs
            in sources_brirs should figure in diffuse_brirs for realism
            purposes. Must have same length as xs_diffuse.
        snr:
            Directional components to diffuse noise signal-to-noise ratio. The
            total energy of the directional noise sources is compared to the
            diffuse noise. The directional sources all have the same level.
        ltas:
            If True, the diffuse noise is equalized to match the LTAS of the
            TIMIT database.

    Returns:
        mixture:
            Output mixture. Shape n_samples*2.
    '''
    if len(xs_sources) != len(brirs_sources):
        raise ValueError(('xs_sources and brirs_sources must have same '
                          'length'))
    if len(xs_diffuse) != len(brirs_diffuse):
        raise ValueError(('xs_diffuse and brirs_diffuse must have same '
                          'length'))
    if xs_sources:
        n_samples = len(xs_sources[0])
    elif xs_diffuse:
        broken = False
        for x_diffuse in xs_diffuse:
            if x_diffuse is not None:
                n_samples = len(x_diffuse)
                broken = True
                break
        if not broken:
            return None
    else:
        return None
    directional_sources = np.zeros((n_samples, 2))
    for x, brir in zip(xs_sources, brirs_sources):
        directional_sources += spatialize(x, brir)
    diffuse_noise = np.zeros((n_samples, 2))
    for x, brir in zip(xs_diffuse, brirs_diffuse):
        if x is not None:
            diffuse_noise += spatialize(x, brir)
    if not (diffuse_noise == 0).all() and ltas:
        ltas = np.load('ltas.npy')
        diffuse_noise = match_ltas(diffuse_noise, ltas)
    if not (directional_sources == 0).all() and not (diffuse_noise == 0).all():
        diffuse_noise, _ = adjust_snr(directional_sources, diffuse_noise, snr)
    return diffuse_noise + directional_sources


def make_mixture(x_target, brir_target, brirs_diffuse, brirs_directional, snr,
                 snr_directional_to_diffuse, xs_diffuse, xs_directional,
                 rms_dB=0, padding=0, reflection_boundary=50e-3, fs=16e3,
                 ltas=False):
    '''
    Make a binaural mixture consisting of a target signal, some diffuse noise
    and some directional noise sources

    Parameters:
        x_target:
            Clean talker monaural signal.
        brir_target:
            Binaural room impulse response used to spatialize the target before
            mixing. This defines the position of the talker in the room.
        brirs_diffuse:
            List of binaural room impulse responses used to create diffuse
            noise.
        brirs_directional:
            List of binaural room impulse responses used to create the
            directional noise sources.
        snr:
            Signal-to-noise ratio, where "signal" refers to the reverberant
            target signal and "noise" refers to the diffuse noise plus the
            directional noise.
        snr_directional_to_diffuse:
            Directional components to diffuse noise signal-to-noise ratio. The
            total energy of the directional noise sources is compared to the
            diffuse noise. The directional sources all have the same level.
        xs_diffuse:
            Clean noise signals used to create diffuse noise using
            brirs_diffuse. All the elements must have same length as x_target.
        xs_directional:
            List of clean directional noise signals to convolve with
            brirs_directional. All the elements must have same length as
            x_target and as the elements of xs_diffuse.
        rms_dB:
            RMS of the total mixture in dB, with unit reference.
        padding:
            Amount of zeros to add before and after the target signal before
            mixing with noise, in seconds.
        reflection_boundary:
            Reflection boundary defining the limit between early and late
            reflections.
        fs:
            Sampling frequency.
        ltas:
            If True, the diffuse noise is equalized to match the LTAS of the
            TIMIT database.

    Returns:
        mixture:
            Reverberant binaural mixture.
        foreground:
            Target signal early reflections. Serves as the target signal in
            the IRM calculation.
        background:
            Target signal late reflections plus diffuse and directional noise
            components. Serves as the noise signal in the IRM calculation.
    '''
    padding = round(padding*fs)
    brir_early, brir_late = split_brir(brir_target, reflection_boundary, fs)
    target_full = spatialize(x_target, brir_target)
    target_early = spatialize(x_target, brir_early)
    target_late = spatialize(x_target, brir_late)
    target_full = zero_pad(target_full, padding, 'both')
    target_early = zero_pad(target_early, padding, 'both')
    target_late = zero_pad(target_late, padding, 'both')
    noise = diffuse_and_directional_noise(xs_directional,
                                          brirs_directional,
                                          xs_diffuse,
                                          brirs_diffuse,
                                          snr_directional_to_diffuse,
                                          ltas)
    if noise is None:
        noise = 0
    else:
        noise, _ = adjust_snr(target_full, noise, snr,
                              slice(padding, len(target_full)-padding))
    mixture = target_full + noise
    foreground = target_early
    background = target_late + noise
    mixture, gain = adjust_rms(mixture, rms_dB)
    foreground *= gain
    background *= gain
    return mixture, foreground, background


class Mixture:
    def __init__(self):
        self.early_target = None
        self.late_target = None
        self.directional_noise = None
        self.diffuse_noise = None

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
        self.early_target = spatialize(x, brir_early)
        self.late_target = spatialize(x, brir_early)
        n_pad = round(pad*fs)
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
