import numpy as np
import scipy.signal

from .utils import zero_pad, fft_freqs


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
    x_left = scipy.signal.oaconvolve(x, brir[:, 0], mode='same')
    x_right = scipy.signal.oaconvolve(x, brir[:, 1], mode='same')
    return np.vstack([x_left, x_right]).T


def spatialize_multi(x, brirs):
    '''
    Sum of convolutions of input signal with multiple BRIRs

    Parameters:
        x:
            Input signal.
        brirs:
            List of BRIRs to convolve the input signal with.

    Returns:
        y:
            Sum of convolutions of x with each BRIR in brirs. Shape
            length(x)*2.
    '''
    y = np.zeros((len(x), 2))
    for brir in brirs:
        y += spatialize(x, brir)
    return y


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
    gain = 10**(-snr/10)*(energy_signal/energy_noise)**0.5
    noise_scaled = noise*gain
    return noise_scaled


def diffuse_and_directional_noise(xs_sources, brirs_sources, x_diffuse,
                                  brirs_diffuse, snrs):
    '''
    Creates a mixture consisting of a set of directional noise sources in
    diffuse noise at given SNRs.

    Parameters:
        x_sources:
            List of clean directional noise signals to convolve with
            brirs_sources. Must all have same length.
        brirs_sources:
            List of BRIRs to convolve each directional noise signal with.
            Must have same length as x_sources.
        x_diffuse:
            Clean noise signal to make diffuse using brirs_diffuse. Must have
            same length as the elements of x_sources.
        brirs_diffuse:
            List of BRIRs to use to create the diffuse noise. Ideally the BRIRs
            in sources_brirs should figure in diffuse_brirs for realism
            purposes.
        snrs:
            List of SNR values for each directional noise source. Should have
            same length as x_sources and brirs_sources

    Returns:
        mixture:
            Output mixture. Shape n_samples*2.
    '''
    if not len(xs_sources) == len(brirs_sources) == len(snrs):
        raise ValueError(('xs_sources, brirs_sources and snrs must have same '
                          'length'))
    diffuse_noise = spatialize_multi(x_diffuse, brirs_diffuse)
    directional_sources = np.zeros((len(diffuse_noise), 2))
    for x, brir, snr in zip(xs_sources, brirs_sources, snrs):
        source = spatialize(x, brir)
        source = adjust_snr(diffuse_noise, source, -snr)
        directional_sources += source
    return diffuse_noise + directional_sources


def make_mixture(x_target, brir_target, brirs_diffuse, brirs_directional, snr,
                 snrs_directional_to_diffuse, x_diffuse, xs_directional,
                 padding=0, reflection_boundary=50e-3, fs=16e3):
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
        snrs_directional_to_diffuse:
            List of ignal-to-noise ratios, where "signal" refers to the
            directional noise and "noise" refers to the diffuse noise.
        x_diffuse:
            Clean noise signal to make diffuse using brirs_diffuse.
        xs_directional:
            List of clean directional noise signals to convolve with
            brirs_directional.
        padding:
            Amount of zeros to add before and after the target signal before
            mixing with noise, in seconds.
        reflection_boundary:
            Reflection boundary defining the limit between early and late
            reflections.
        fs:
            Sampling frequency.

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
                                          x_diffuse,
                                          brirs_diffuse,
                                          snrs_directional_to_diffuse)
    noise = adjust_snr(target_full, noise, snr,
                       slice(padding, len(target_full)-padding))
    mixture = target_full + noise
    foreground = target_early
    background = target_late + noise
    return mixture, foreground, background
