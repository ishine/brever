import numpy as np
import scipy.fftpack

from .utils import standardize, frame
from .filters import filt


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


def ild(x, filtered=False, filt_kwargs=None, framed=False, frame_kwargs=None):
    '''
    Interaural level difference.

    Parameters:
        x:
            Input signal.
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
        ild:
            Interaural level difference. Size n_frames*n_filters.
    '''
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    energy = np.sum(x**2, axis=1)
    return 10*np.log10(energy[:, :, 1]/energy[:, :, 0])


def itd(x, filtered=False, filt_kwargs=None, framed=False, frame_kwargs=None):
    '''
    Interaural time difference.

    Parameters:
        x:
            Input signal.
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
        itd:
            Interaural time difference. Size n_frames*n_filters.
    '''
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    CCF, lags = ccf(x[:, :, :, 1], x[:, :, :, 0], max_lag=16,
                    negative_lags=True, method='fft', axis=1, normalize=True)
    return lags[CCF.argmax(axis=1)]


def ic(x, filtered=False, filt_kwargs=None, framed=False, frame_kwargs=None):
    '''
    Interaural coherence.

    Parameters:
        x:
            Input signal.
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
        ic:
            Interaural coherence. Size n_frames*n_filters.
    '''
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    CCF, lags = ccf(x[:, :, :, 1], x[:, :, :, 0], max_lag=16,
                    negative_lags=True, method='fft', axis=1, normalize=True)
    return CCF.max(axis=1)


def itd_ic(x, filtered=False, filt_kwargs=None, framed=False,
           frame_kwargs=None):
    '''
    Interaural level difference and interaural coherence. Calculating these
    together speeds up dataset generation because they both come from the
    cross-correlation function.

    Parameters:
        x:
            Input signal.
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
        itd-ic:
            Interaural time difference and interaural coherence stacked
            together. Size n_frames*(n_filters*2).
    '''
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    CCF, lags = ccf(x[:, :, :, 1], x[:, :, :, 0], max_lag=16,
                    negative_lags=True, method='fft', axis=1, normalize=True)
    ITD = lags[CCF.argmax(axis=1)]
    IC = CCF.max(axis=1)
    return np.hstack([ITD, IC])


def mfcc(x, n_mfcc=13, dct_type=2, norm='ortho', filtered=False,
         filt_kwargs=None, framed=False, frame_kwargs=None):
    '''
    Mel-frequency cepstral coefficients. DC term is not returned.

    Parameters:
        x:
            Input signal.
        n_mfcc:
            Number of MFCCs to return, DC term not included.
        dct_type:
            Discrete cosine transform type.
        norm:
            If dct_type is 2 or 3, setting norm='ortho' uses an ortho-normal
            DCT basis.
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
        mfcc:
            Mel-frequency cepstral coefficients. Size n_frames*n_mfcc.
    '''
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    x = x.mean(axis=-1)  # average channels
    energy = x**2  # get energy
    energy = energy.mean(axis=1)  # average each frame
    log_energy = np.log(energy)
    mfcc = scipy.fftpack.dct(log_energy, axis=1, type=dct_type, norm=norm)
    return mfcc[:, 1:n_mfcc+1]


def pdf(x, filtered=False, filt_kwargs=None, framed=False, frame_kwargs=None):
    '''
    Probability density function estimate.

    Parameters:
        x:
            Input signal.
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
        pdf:
            Probability density function estimate. Size n_frames*n_filters.
    '''
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    x = x.mean(axis=-1)  # average channels
    energy = x**2  # get energy
    energy = energy.mean(axis=1)  # average each frame
    pdf = energy/energy.sum(axis=1, keepdims=True)
    return pdf


def _check_input(x, filtered=False, filt_kwargs=None, framed=False,
                 frame_kwargs=None):
    if framed and not filtered:
        raise ValueError('framed cannot be True if filtered is False, since '
                         'framing must be done after filtering')
    if filt_kwargs is None:
        filt_kwargs = {}
    if frame_kwargs is None:
        frame_kwargs = {}
    if x.shape[-1] != 2:
        raise ValueError(('the last dimension of x must be the number of '
                          'channels which must be 2'))
    if not filtered:
        if x.ndim != 2:
            raise ValueError(('when filtered is False, x should be '
                              '2-dimensional with size n_samples*2'))
        x, _ = filt(x, **filt_kwargs)
    if not framed:
        if x.ndim != 3:
            raise ValueError(('when filtered is True and framed is False, x '
                              'should be 3-dimensional with size '
                              'n_samples*n_filters*2'))
        x = frame(x, **frame_kwargs)
    if x.ndim != 4:
        raise ValueError(('when filtered is True and framed is True, x should '
                          'be 4-dimensional with size '
                          'n_frames*frame_length*n_filters*2'))
    return x
