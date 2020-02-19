import numpy as np

from .utils import standardize, frame


def ccf(x, y, method='convolve', max_lag=40, negative_lags=False, axis=0,
        normalize=True):
    '''
    Cross-correlation function or cross-covariance function.

    Parameters:
        x:
            First time series i.e. the one the delay is applied to.
        y:
            Second time series.
        method:
            'convolve' or 'fft'.
            - If 'convolve', the cross-correlation is calculated as usually
            done in time series analysis, i.e. purely in time domain by
            summation of products up to lag max_lag. This implementation
            matches the equivalent funuction in R. Currently only supports one
            dimensional inputs.
            - If 'fft', the calculation is done by multiplication in the
            frequency domain. The inputs should have same length and ideally a
            power of two. Supports multidimensional input.
        max_lag:
            Maximum lag to compute the cross-correlation for.
        negative_lags:
            If True, negative lags are also calculated i.e. y is also delayed.
        axis:
            Axis along which to compute the cross-correlation. Currently not
            supported together with 'convolve' method.
        normalize:
            If True, the inputs are standardized before convolving. This
            ensures the output values are between -1 and 1, and thus a true
            cross-correlation function is calculated. If False, the output ends
            up being the cross-covariance function instead.

    Returns:
        CCF:
            Cross-correlation values.
        lags:
            Lag vector in samples. Same length as CCF.
    '''
    # TODO: do gcc instead, since the interaural coherance is taking very small
    # values
    if normalize:
        x = standardize(x, axis=axis)
        y = standardize(y, axis=axis)
    if method == 'convolve':
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(('inputs must be one-dimensional when method is '
                              'convolve'))
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


def ild(x_filt, frame_length=512, hop_length=256):
    '''
    ILD from the output of a filterbank.

    Parameters:
        x_filt:
            Signal decomposed by a filterbank. Size n_samples*n_filters*2.
        frame_length:
            Frame length in samples.
        hop_length:
            Frame shift in samples.

    Returns:
        ILD:
            ILD. Size n_frames*n_filters.
    '''
    if x_filt.ndim != 3 or x_filt.shape[2] != 2:
        raise ValueError('x_filt should have shape n_samples*n_filters*2')
    frames = frame(x_filt, frame_length, hop_length)
    energy = np.sum(frames**2, axis=1)
    ILD = 10*np.log10(energy[:, :, 1]/energy[:, :, 0])
    return ILD


def itd(x_filt, frame_length=512, hop_length=256):
    '''
    ITD from the output of a filterbank.

    Parameters:
        x_filt:
            Signal decomposed by a filterbank. Size n_samples*n_filters*2.
        frame_length:
            Frame length in samples.
        hop_length:
            Frame shift in samples.

    Returns:
        ITD:
            ITD. Size n_frames*n_filters.
    '''
    if x_filt.ndim != 3 or x_filt.shape[2] != 2:
        raise ValueError('x_filt should have shape n_samples*n_filters*2')
    frames = frame(x_filt, frame_length, hop_length)
    n_frames, _, n_filters, _ = frames.shape
    CCF, lags = ccf(frames[:, :, :, 1], frames[:, :, :, 0], max_lag=16,
                    negative_lags=True, method='fft', axis=1, normalize=True)
    ITD = lags[CCF.argmax(axis=1)]
    return ITD


def ic(x_filt, frame_length=512, hop_length=256):
    '''
    Interautal coherence from the output of a filterbank.

    Parameters:
        x_filt:
            Signal decomposed by a filterbank. Size n_samples*n_filters*2.
        frame_length:
            Frame length in samples.
        hop_length:
            Frame shift in samples.

    Returns:
        ILD:
            ILD. Size n_frames*n_filters.
    '''
    if x_filt.ndim != 3 or x_filt.shape[2] != 2:
        raise ValueError('x_filt should have shape n_samples*n_filters*2')
    frames = frame(x_filt, frame_length, hop_length)
    n_frames, _, n_filters, _ = frames.shape
    CCF, lags = ccf(frames[:, :, :, 1], frames[:, :, :, 0], max_lag=16,
                    negative_lags=True, method='fft', axis=1, normalize=True)
    IC = CCF.max(axis=1)
    return IC
