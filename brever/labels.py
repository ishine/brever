import numpy as np

from .utils import frame
from .filters import filt


def irm(target, noise, filtered=False, filt_kwargs=None, framed=False,
        frame_kwargs=None):
    '''
    Ideal ratio mask. If the input signals are multichannel, the channels are
    averaged to create monaural signals.

    Parameters:
        target:
            Target signal.
        noise:
            Noise signal.
        filtered:
            If True, the input signals are assumed to be already filtered. They
            should then have size n_samples*n_filters*2.
        filt_kwargs
            Keyword arguments passed to filters.filt if filtered is
            False.
        framed:
            If True, the input signals are assumed to be already framed. They
            should then have size n_frames*frame_length*n_filters*2.
        frame_kwargs:
            Keyword arguments passed to utils.frame if framed is False.

    Returns:
        IRM:
            Ideal ratio mask.
    '''
    target = _check_input(target, filtered, filt_kwargs, framed, frame_kwargs)
    noise = _check_input(noise, filtered, filt_kwargs, framed, frame_kwargs)
    target = target.mean(axis=-1)
    noise = noise.mean(axis=-1)
    energy_target = np.mean(target**2, axis=1)
    energy_noise = np.mean(noise**2, axis=1)
    energy_mix = energy_target + energy_noise
    energy_mix[energy_mix == 0] = 1
    IRM = (energy_target/energy_mix)**0.5
    return IRM


def _check_input(x, filtered=False, filt_kwargs=None, framed=False,
                 frame_kwargs=None):
    '''
    Checks input before label calculation and transforms it if necesarry.

    Parameters:
        x:
            Input signal.
        filtered:
            Wether the input signal is already filtered or not.
        filt_kwargs:
            If filtered is False, the input signal is filtered using
            filt_kwargs as keyword arguments.
        framed:
            Wether the input signal is already framed or not.
        frame_kwargs:
            If framed is False, the input signal is framed using
            frame_kwargs as keyword arguments.
    '''
    if filt_kwargs is None:
        filt_kwargs = {}
    if frame_kwargs is None:
        frame_kwargs = {}
    if x.shape[-1] != 2:
        raise ValueError(('the last dimension of x must be the number of '
                          'channels which must be 2'))
    if not filtered:
        if x.ndim != 2:
            raise ValueError(('x should be 2-dimensional with size '
                              'n_samples*2.'))
        x, _ = filt(x, **filt_kwargs)
    if not framed:
        if x.ndim != 3:
            raise ValueError(('x should be 3-dimensional with size '
                              'n_samples*n_filters*2.'))
        x = frame(x, **frame_kwargs)
    if x.ndim != 4:
        raise ValueError(('x should be 4-dimensional with size '
                          'n_frames*frame_length*n_filters*2.'))
    return x
