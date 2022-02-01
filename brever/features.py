import numpy as np
import scipy.fft

from .utils import standardize, pad

eps = np.finfo(float).eps


def ccf(x, y, method='fft', max_lag=16, negative_lags=True, axis=-1,
        normalize=True):
    """
    Cross-correlation function.

    Calculates the cross-correlation function or cross-covariance function from
    two input signals.

    Parameters
    ----------
    x : array_like
        First time series i.e. the one the delay is applied to.
    y : array_like
        Second time series.
    method : {'convolve', 'fft'}, optional
        Calculation method. Default is `'fft'`.
        ``fft``
            The calculation is done by multiplication in the frequency domain.
            The inputs should have same shape. Supports multidimensional input.
        ``convolve``
            The cross-correlation is in time series analysis, i.e. purely in
            time domain by summation of products up to lag `max_lag`. This
            implementation matches the equivalent function in R. Currently
            only supports one-dimensional inputs.
    max_lag : int, optional
        Maximum lag the cross-correlation is calculated for. Default is 16.
    negative_lags : bool, optional
        If `True`, negative lags are also calculated, i.e. `y` is also delayed.
        Default is `True`.
    axis : int, optional
        Axis along which to compute the cross-correlation. Default is -1.
    normalize : bool, optional
        If `True`, the inputs are standardized before convolving. This
        ensures the output values lie between -1 and 1, and thus a true
        cross-correlation function is calculated. If `False`, the output is
        instead the cross-covariance function. Default is `True`.

    Returns
    -------
    CCF : array_like
        Cross-correlation values.
    lags : array_like
        Lag vector in samples. Same length as CCF.
    """
    if normalize:
        x = standardize(x, axis=axis)
        y = standardize(y, axis=axis)
    if method == 'convolve':
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError('inputs must be one-dimensional when method is '
                             'convolve')
        CCF = np.zeros(min(len(y), max_lag+1))
        n = max(len(x), len(y))
        for i in range(len(CCF)):
            y_advanced = y[i:]
            m = min(len(x), len(y_advanced))
            CCF[i] = np.sum(x[:m]*y_advanced[:m])
        CCF /= n
        lags = np.arange(len(CCF))
        if negative_lags:
            CCF_neg, lags_neg = ccf(y, x, method='convolve', max_lag=max_lag,
                                    negative_lags=False)
            CCF = np.hstack((CCF, CCF_neg[1:]))
            lags = np.hstack((lags, -lags[1:]))
    elif method == 'fft':
        n = x.shape[axis]
        X = np.fft.fft(x, axis=axis)
        Y = np.fft.fft(y, axis=axis)
        CCF = 1/n*np.fft.ifft(np.conj(X)*Y, axis=axis).real
        # CCF = np.roll(CCF, (n+1)//2)
        lags = np.arange(n)
        mask = lags > n/2
        lags[mask] = lags[mask] - n
        mask = (lags <= max_lag) & (lags >= -max_lag)
        if not negative_lags:
            mask = mask & (lags >= 0)
        CCF = np.compress(mask, CCF, axis=axis)
        lags = lags[mask]
    else:
        return ValueError('method should be either convolve or fft')
    return CCF, lags


def ild(x, time_axis=-1, channel_axis=-3):
    """
    Interaural level difference.

    Calculates the interaural level difference between two channels.

    Parameters
    ----------
    x : array_like
        Input signal.
    time_axis: int, optional
        Time axis in input array. Default is -1.
    channel_axis : int, optional
        Channel axis in input array. The size of `x` along `axis` must be 2.
        Default is -3.

    Returns
    -------
    ild : array_like
        Interaural level difference.
    """
    assert x.shape[channel_axis] == 2
    energy = np.sum(x**2, axis=time_axis, keepdims=True) + eps
    left = np.take(energy, [0], axis=channel_axis)
    right = np.take(energy, [1], axis=channel_axis)
    ild = 10*np.log10(left/right)
    return ild.squeeze(axis=(time_axis, channel_axis))


def itd(x, time_axis=-1, channel_axis=-3):
    """
    Interaural time difference.

    Calculates the interaural time difference between two channels.

    Parameters
    ----------
    x : array_like
        Input signal.
    time_axis: int, optional
        Time axis in input array. Default is -1.
    channel_axis : int, optional
        Channel axis in input array. The size of `x` along `axis` must be 2.
        Default is -3.

    Returns
    -------
    itd : array_like
        Interaural time difference.
    """
    assert x.shape[channel_axis] == 2
    left = np.take(x, [0], axis=channel_axis)
    right = np.take(x, [1], axis=channel_axis)
    CCF, lags = ccf(left, right, axis=time_axis)
    itd = lags[CCF.argmax(axis=time_axis, keepdims=True)]
    return itd.squeeze(axis=(time_axis, channel_axis))


def ic(x, time_axis=-1, channel_axis=-3):
    """
    Interaural coherence.

    Calculates the interaural coherence between two channels.

    Parameters
    ----------
    x : array_like
        Input signal.
    time_axis: int, optional
        Time axis in input array. Default is -1.
    channel_axis : int, optional
        Channel axis in input array. The size of `x` along `axis` must be 2.
        Default is -3.

    Returns
    -------
    ic : array_like
        Interaural coherence.
    """
    assert x.shape[channel_axis] == 2
    left = np.take(x, [0], axis=channel_axis)
    right = np.take(x, [1], axis=channel_axis)
    CCF, lags = ccf(left, right, axis=time_axis)
    ic = CCF.max(axis=time_axis, keepdims=True)
    return ic.squeeze(axis=(time_axis, channel_axis))


def fbe(x, normalize=False, compression='none', dct=False, n_dct=14,
        dct_type=2, dct_norm='ortho', return_dc=False, return_deltas=True,
        return_double_deltas=True, time_axis=-1, frame_axis=-2,
        channel_axis=-3, frequency_axis=0):
    """
    Filterbank energies.

    Calculates the energy in each time-frequency unit. Supports a series of
    compression and normalization options to obtain MFCC or PDF features.

    Parameters
    ----------
    x : array_like
        Input signal.
    normalize : bool, optional
        Whether to normalize along frequency axis. Default is `False`.
    compression : {'log', 'cubic', 'none'}, optional
        Compression type. Default is `'none'`.
    dct : bool, optional
        Wheter to apply DCT compression along the frequency axis. Default is
        `False`.
    n_dct : int, optional
        Number of DCT coefficients to return, including DC term. Ignored if
        `dct` is `False`. Default is 14.
    dct_type : {1, 2, 3, 4}, optional
        Type of DCT. Ignored if `dct` is `False`. Default is 2.
    dct_norm : {'backward', 'ortho', 'forward'}, optional
        Normalization mode for the DCT. Ignored if `dct` is `False`. Default is
        `'ortho'`.
    return_dc : bool, optional
        Whether to return the DC term. Ignored if `dct` is `False`. Default is
        `False`.
    return_deltas : bool or None, optional
        Whether to return first order difference along the frame axis. Ignored
        if `dct` is `False`. Default is `True`.
    return_double_deltas : bool or None, optional
        Whether to return second order difference along the frame axis. Ignored
        if `dct` is `False`. Default is `True`.
    time_axis: int, optional
        Time axis in input array. Default is -1.
    frame_axis : int, optional
        Frame axis in input array. Default is -2.
    channel_axis : int, optional
        Channel axis in input array. The size of `x` along `axis` must be 2.
        Default is -3.
    frequency_axis : int, optional
        Frequency axis in input array. Default is 0.

    Returns
    -------
    fbe : array_like
        Filterbank energies.
    """
    assert x.shape[channel_axis] == 2
    # calculate energy
    out = np.sum(x**2, axis=(time_axis, channel_axis), keepdims=True)
    # normalize
    if normalize:
        out /= out.sum(axis=frequency_axis, keepdims=True) + eps
    # compress
    if compression == 'log':
        out = np.log(out + eps)
    elif compression == 'cubic':
        out = out**(1/3)
    elif compression != 'none':
        raise ValueError('compression must be log, cubic or none, got '
                         f'{compression}')
    # apply dct
    if dct:
        out = scipy.fft.dct(out, axis=frequency_axis, type=dct_type,
                            norm=dct_norm)
        if return_dc:
            out = np.take(out, range(n_dct), axis=frequency_axis)
        else:
            out = np.take(out, range(1, n_dct), axis=frequency_axis)
        present = out
        if return_deltas:
            diff = np.diff(present, n=1, axis=frame_axis)
            diff = pad(diff, n=1, axis=frame_axis, where='left')
            out = np.concatenate((out, diff), axis=frequency_axis)
        if return_double_deltas:
            diff = np.diff(present, n=2, axis=frame_axis)
            diff = pad(diff, n=2, axis=frame_axis, where='left')
            out = np.concatenate((out, diff), axis=frequency_axis)
    return out.squeeze(axis=(time_axis, channel_axis))


class FeatureExtractor:

    feature_map = {
        'ild': ild,
        'itd': itd,
        'ic': ic,
        'fbe': fbe,
        'logfbe': lambda x: fbe(x, compression='log'),
        'cubicfbe': lambda x: fbe(x, compression='cubic'),
        'pdf': lambda x: fbe(x, normalize=True),
        'logpdf': lambda x: fbe(x, normalize=True, compression='log'),
        'cubicpdf': lambda x: fbe(x, normalize=True, compression='cubic'),
        'mfcc': lambda x: fbe(x, compression='log', dct=True),
        'cubicmfcc': lambda x: fbe(x, compression='cubic', dct=True),
        'pdfcc': lambda x: fbe(x, normalize=True, compression='log', dct=True),
    }

    def __init__(self, features):
        self.features = features
        self.indices = None

    def __call__(self, x):
        output = []
        self.indices = {}
        i_start = 0
        for feature in self.features:
            feature_func = self.feature_map[feature]
            data = feature_func(x)
            output.append(feature_func(x))
            i_end = i_start + len(data)
            self.indices[feature] = (i_start, i_end)
            i_start = i_end
        return np.concatenate(output, axis=0)
