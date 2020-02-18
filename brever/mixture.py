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


def make(target, brir, brirs, snr, padding=0):
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
            Number of zeros to add before and after the target signal before
            mixing with noise, in samples (not seconds).

    Returns:
        mix:
            Reverberant Binaural mixture.
        target:
            Reverberant target signal.
        noise:
            Diffuse noise signal.
    '''
    target_reverb = spatialize(target, brir)
    target_reverb = zero_pad(target_reverb, padding, 'both')
    n_samples = len(target_reverb) + padding*2
    noise = diffuse_noise(brirs, n_samples)
    energy_noise = np.sum(noise[padding:n_samples-padding]**2)
    energy_signal = np.sum(target_reverb**2)
    noise *= 10**(-snr/10)*(energy_signal/energy_noise)**0.5
    return target_reverb+noise, target_reverb, noise
