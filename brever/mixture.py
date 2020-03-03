import numpy as np
import scipy.signal

from .utils import zero_pad


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


def diffuse_noise(brirs, n_samples):
    '''
    Create diffuse white Gaussian noise using a set of binaural room impulse
    responses.

    Parameters:
        brirs:
            List of binaural room impulse responses.
        n_samples:
            Number of samples of noise to generate.

    Returns:
        noise:
            Diffuse binaural noise.
    '''
    noise = np.zeros((n_samples, 2))
    for brir in brirs:
        noise += spatialize(np.random.randn(n_samples), brir)
    return noise


def split_brir(brir, reflection_boundary=10e-3, max_itd=1e-3, fs=16e3):
    '''
    Splits a BRIR into a direct or early reflections component and a reverb or
    late reflections component.

    Parameters:
        brir:
            Input BRIR.
        reflection_boundary:
            Reflection boundary defining the limit between early and late
            reflections.
        max_itd:
            Maximum interaural time difference. Used to compare the locations
            of the peak in each channel and correct if necessary.
        fs:
            Sampling frequency.

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


def make(target, brir, brirs, snr, padding=0, reflection_boundary=10e-3,
         max_itd=1e-3, fs=16e3):
    '''
    Make a binaural mixture consisting of a target signal and diffuse noise
    at a given SNR.

    Parameters:
        target:
            Talker monaural signal.
        brir:
            Binaural room impulse response used to spatialize the target before
            mixing. This defines the position of the talker in the room.
        brirs:
            List of binaural room impulse responses used ot create diffuse
            noise. brir should ideally figure in this list.
        snr:
            Signal-to-noise ratio.
        padding:
            Amount of zeros to add before and after the target signal before
            mixing with noise, in seconds.
        reflection_boundary:
            Reflection boundary defining the limit between early and late
            reflections.
        max_itd:
            Maximum interaural time difference. Used to compare the locations
            of the peak in each channel and correct if necessary.
        fs:
            Sampling frequency.

    Returns:
        mix:
            Reverberant Binaural mixture.
        target:
            Reverberant target signal, consisting of both direct sound and late
            reflections. We have target_reverb = target_early + target_late.
        target_early:
            Direct contribution of the target signal. Useful for IRM
            calculation.
        target_late:
            Late contribution of the target signal. Useful for IRM calculation.
        noise:
            Diffuse noise signal.
    '''
    padding = round(padding*fs)
    target_reverb = spatialize(target, brir)
    target_reverb = zero_pad(target_reverb, padding, 'both')
    n_samples = len(target_reverb) + round(padding*2)
    noise = diffuse_noise(brirs, n_samples)
    energy_noise = np.sum(noise[padding:n_samples-padding]**2)
    energy_signal = np.sum(target_reverb**2)
    noise *= 10**(-snr/10)*(energy_signal/energy_noise)**0.5
    mix = target_reverb+noise
    brir_direct, brir_late = split_brir(brir, reflection_boundary, max_itd, fs)
    target_early = spatialize(target, brir_direct)
    target_late = spatialize(target, brir_late)
    return mix, target_reverb, target_early, target_late, noise
