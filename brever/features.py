import numpy as np
import scipy.signal

from .utils import standardize, frame


def ccf(x, y, method='convolve', max_lag=40, negative_lags=False):
    '''
    Cross-correlation function. Output values are between -1 and 1, i.e. the
    cross-covariance is normalized by the standard deviations of the input
    series.

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
            matches the equivalent funuction in R.
            - If 'fft', the calculation is done by multiplication in the
            frequency domain. The inputs should have same length and ideally a
            power of two. Supports multichannel input.
        max_lag:
            Maximum lag to compute the cross-correlation for.
        negative_lags:
            If True, negative lags are also calculated i.e. y is also delayed.

    Returns:
        CCF:
            Cross-correlation values.
        lags:
            Lag vector in samples. Same length as CCF.
    '''
    x = standardize(x)
    y = standardize(y)
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
            lags = np.hstack((lags, lags[1:]))
    elif method == 'fft':
        n = len(x)
        X = np.fft.fft(x, axis=0)
        Y = np.fft.fft(y, axis=0)
        CCF = 1/n*np.fft.ifft(np.conj(X)*Y, axis=0).real
        # CCF = np.roll(CCF, (n+1)//2)
        lags = np.arange(n)
        mask = lags > n/2
        lags[mask] = lags[mask] - n
        mask = (lags <= max_lag) & (lags >= -max_lag)
        if not negative_lags:
            mask = mask & (lags >= 0)
        CCF = CCF[mask]
        lags = lags[mask]
    else:
        return ValueError('method should be either convolve or fft')
    return CCF, lags


def itd(x_filt, frame_length=512, hop_length=256):
    '''
    ITD from the output of a gammatone filterbank.

    Parameters:
        x_filt:
            Signal decomposed by a gammatone filterbank. Size
            n_samples*n_filters*2.
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
    frames_left = frame(x_filt[:, :, 0], frame_length, hop_length)
    frames_right = frame(x_filt[:, :, 1], frame_length, hop_length)
    n_frames, _, n_filters = frames_left.shape
    ITD = np.zeros((n_frames, n_filters))
    for i in range(n_frames):
        frame_left = frames_left[i, :, :]
        frame_right = frames_right[i, :, :]
        CCF, lags = ccf(frame_right, frame_left, max_lag=16,
                        negative_lags=True, method='fft')
        ITD[i, :] = lags[np.argmax(CCF, axis=0)]
    return ITD


def ild(x_filt, frame_length=512, hop_length=256):
    '''
    ILD from the output of a gammatone filterbank.

    Parameters:
        x_filt:
            Signal decomposed by a gammatone filterbank. Size
            n_samples*n_filters*2.
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
    frames_left = frame(x_filt[:, :, 0], frame_length, hop_length)
    frames_right = frame(x_filt[:, :, 1], frame_length, hop_length)
    n_frames, _, n_filters = frames_left.shape
    ILD = np.zeros((n_frames, n_filters))
    for i in range(n_frames):
        frame_left = frames_left[i, :, :]
        frame_right = frames_right[i, :, :]
        energy_left = np.sum(frame_left**2, axis=0)
        energy_right = np.sum(frame_right**2, axis=0)
        ILD[i, :] = 10*np.log10(energy_right/energy_left)
    return ILD


def ic(x_filt, tau=0.1, fs=16e3, frame_length=512, hop_length=256):
    alpha = np.exp(-hop_length/(tau*fs))
    b = [1 - alpha]
    a = [1, -alpha]
    phi_ll = x_filt[:, :, 0]**2
    phi_rr = x_filt[:, :, 1]**2
    phi_lr = x_filt[:, :, 0]*x_filt[:, :, 1]
    phi_ll = scipy.signal.lfilter(b, a, phi_ll, axis=0)
    phi_rr = scipy.signal.lfilter(b, a, phi_rr, axis=0)
    phi_lr = scipy.signal.lfilter(b, a, phi_lr, axis=0)
    IC = phi_lr/(phi_ll*phi_rr)**0.5
    IC = frame(IC, frame_length, hop_length)
    IC = np.mean(IC, axis=1)
    return IC
