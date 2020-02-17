import numpy as np

from .utils import frame
from .timefreq import gammatone_filt


def irm(target, noise, frame_length=512, hop_length=256):
    '''
    Ideal ratio mask. If the input signals are multichannel, the channels are
    averaged to create monaural signals.

    Parameters:
        target:
            Target signal.
        noise:
            Noise signal.
        frame_length:
            Frame length in samples.
        hop_length:
            Frame shift in samples.

    Returns:
        IRM:
            Ideal ratio mask.
    '''
    # TODO: add option to chose tf_analysis stage, either gammatone or stft
    if target.ndim == 2:
        target = target.mean(axis=-1)
    if noise.ndim == 2:
        noise = noise.mean(axis=-1)
    target, _ = gammatone_filt(target)
    noise, _ = gammatone_filt(noise)
    target = frame(target, frame_length, hop_length)
    noise = frame(noise, frame_length, hop_length)
    energy_target = np.mean(target**2, axis=1)
    energy_noise = np.mean(noise**2, axis=1)
    IRM = ((energy_target + 1e-10)/(energy_target + energy_noise + 1e-10))**0.5
    return IRM
