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
            Binaural audio signal.
    '''
    x_left = scipy.signal.oaconvolve(x, brir[:, 0], mode='same')
    x_right = scipy.signal.oaconvolve(x, brir[:, 1], mode='same')
    return np.vstack([x_left, x_right]).T


def diffuse(x, brirs):
    output = np.zeros((len(x), 2))
    for brir in brirs:
        output += spatialize(x, brir)
    return output


def colored_noise(color, n_samples):
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


def diffuse_noise(brirs, n_samples, color='white'):
    '''
    Create diffuse colored noise using a set of binaural room impulse
    responses.

    Parameters:
        brirs:
            List of binaural room impulse responses.
        n_samples:
            Number of samples of noise to generate.
        color:
            Noise color.

    Returns:
        noise:
            Diffuse binaural noise.
    '''
    x = colored_noise(color, n_samples)
    return diffuse(x, brirs)


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


def split_and_spatialize(x, brir, reflection_boundary=50e-3, fs=16e3):
    brir_direct, brir_late = split_brir(brir, reflection_boundary, fs)
    x_early = spatialize(x, brir_direct)
    x_late = spatialize(x, brir_late)
    return x_early, x_late


def adjust_snr(signal, noise, snr, slice_=None):
    if slice_ is None:
        slice_ = np.s_[:]
    energy_signal = np.sum(signal[slice_]**2)
    energy_noise = np.sum(noise[slice_]**2)
    gain = 10**(-snr/10)*(energy_signal/energy_noise)**0.5
    noise = noise*gain
    return noise


def diffuse_and_directional_noise(sources_colors, sources_brirs, diffuse_color,
                                  diffuse_brirs, snrs, n_samples):
    noise = diffuse_noise(diffuse_brirs, n_samples, diffuse_color)
    sources = np.zeros((n_samples, 2))
    for color, brir, snr in zip(sources_colors, sources_brirs, snrs):
        source = colored_noise(color, n_samples)
        source = spatialize(source, brir)
        source = adjust_snr(noise, source, -snr)
        sources += source
    return noise + sources


def make_mixture(target, brir_target, brirs_diffuse, brirs_directional, snr,
                 snrs_directional_to_diffuse, color_diffuse,
                 colors_directional, padding=0, reflection_boundary=50e-3,
                 fs=16e3):
    '''
    Make a binaural mixture consisting of a target signal, some diffuse noise
    and some directional noise sources

    Parameters:
        target:
            Talker monaural signal.
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
        color_diffuse:
            Noise color for the diffuse noise.
        colors_directional:
            List of noise colors for the directional noise sources
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
    target_reverb = spatialize(target, brir_target)
    target_reverb = zero_pad(target_reverb, padding, 'both')
    n_samples = len(target_reverb)
    noise = diffuse_and_directional_noise(colors_directional,
                                          brirs_directional, color_diffuse,
                                          brirs_diffuse,
                                          snrs_directional_to_diffuse,
                                          n_samples)
    noise = adjust_snr(target_reverb, noise, snr,
                       slice(padding, n_samples-padding))
    target_early, target_late = split_and_spatialize(target, brir_target,
                                                     reflection_boundary, fs)
    mixture = target_reverb + noise
    foreground = target_early
    background = target_late + noise
    return mixture, foreground, background
