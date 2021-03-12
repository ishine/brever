import numpy as np
import scipy.fftpack

from .utils import standardize, frame
from .filters import filt


def ccf(x, y, method='convolve', max_lag=40, negative_lags=False, axis=0,
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
        Calculation method. Default is `'convolve'`.
        ``convolve``
            The cross-correlation is calculated as usually done in time series
            analysis, i.e. purely in time domain by summation of products up to
            lag `max_lag`. This implementation matches the equivalent function
            in R. Currently only supports one-dimensional inputs.
        ``fft``
            The calculation is done by multiplication in the frequency domain.
            The inputs should have same length and ideally a power of two.
            Supports multidimensional input.
    max_lag : int, optional
        Maximum lag up to which to compute the cross-correlation for. Default
        is 40.
    negative_lags : bool, optional
        If `True`, negative lags are also calculated i.e. `y` is also delayed.
        Default is `False`.
    axis : int, optional
        Axis along which to compute the cross-correlation. Currently not
        supported together with `'convolve'` method. Default is 0.
    normalize : bool, optional
        If `True`, the inputs are standardized before convolving. This
        ensures the output values lie between -1 and 1, and thus a true
        cross-correlation function is calculated. If `False`, the output ends
        up being the cross-covariance function instead. Default is `True`.

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
    """
    Interaural level difference.

    Calculates the interaural level difference in each time-frequency unit.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.

    Returns
    -------
    ild : array_like
        Interaural level difference. Shape `(n_frames, n_filters)`
    """
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    energy = np.sum(x**2, axis=1) + np.nextafter(0, 1)
    return 10*np.log10(energy[:, :, 1]/energy[:, :, 0])


def itd(x, filtered=False, filt_kwargs=None, framed=False, frame_kwargs=None):
    """
    Interaural time difference.

    Calculates the interaural time difference in each time-frequency unit.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.

    Returns
    -------
    itd : array_like
        Interaural time difference. Shape `(n_frames, n_filters)`
    """
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    CCF, lags = ccf(x[:, :, :, 1], x[:, :, :, 0], max_lag=16,
                    negative_lags=True, method='fft', axis=1, normalize=True)
    return lags[CCF.argmax(axis=1)]


def ic(x, filtered=False, filt_kwargs=None, framed=False, frame_kwargs=None):
    """
    Interaural coherence.

    Calculates the interaural coherence difference in each time-frequency unit.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.

    Returns
    -------
    ic : array_like
        Interaural coherence. Shape `(n_frames, n_filters)`
    """
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    CCF, lags = ccf(x[:, :, :, 1], x[:, :, :, 0], max_lag=16,
                    negative_lags=True, method='fft', axis=1, normalize=True)
    return CCF.max(axis=1)


def itd_ic(x, filtered=False, filt_kwargs=None, framed=False,
           frame_kwargs=None):
    """
    Interaural level difference and interaural coherence.

    Calculates both the interaural level difference and the interaural
    coherence. Calculating these together speeds up the dataset generation
    since they both use the cross-correlation function.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.

    Returns
    -------
    itd-ic : array_like
        Interaural time difference and interaural coherence stacked together.
        Shape `(n_frames, n_filters*2)`.
    """
    x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
    CCF, lags = ccf(x[:, :, :, 1], x[:, :, :, 0], max_lag=16,
                    negative_lags=True, method='fft', axis=1, normalize=True)
    ITD = lags[CCF.argmax(axis=1)]
    IC = CCF.max(axis=1)
    return np.hstack((ITD, IC))


def logmfcc(x, n_mfcc=13, dct_type=2, norm='ortho', filtered=False,
            filt_kwargs=None, framed=False, frame_kwargs=None, energy=None):
    """
    Mel-frequency cepstral coefficients.

    Calculates the mel-frequency cepstral coefficients (MFCC). DC term is not
    returned. Deltas and double deltas are also returned. Uses logarithmic
    compression.

    Parameters
    ----------
    x : array_like
        Input signal.
    n_mfcc : int, optional
        Number of MFCCs to return, DC term not included. Default is 13.
    dct_type : {1, 2, 3, 4}, optional
        Type of the discrete cosine transform (DCT). See the `np.fft.dct`
        documentation. Default is 2.
    norm : {'backward', 'ortho', 'forward'}, optional
        Normalization mode of the discrete cosine transform (DCT). See the
        `np.fft.dct` documentation. Default is `'ortho'`.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.
    energy : array_like
        Pre-computed filterbank energies. Shape `(n_samples, n_filters)`.
        Default is None.

    Returns
    -------
    mfcc : array_like
        Mel-frequency cepstral coefficients. Shape `(n_frames, n_mfcc*3)`.
    """
    if energy is None:
        x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
        x = x.mean(axis=-1)  # average channels
        energy = x**2  # get energy
        energy = energy.mean(axis=1)  # average each frame
    energy_comp = np.log(energy + np.nextafter(0, 1))
    mfcc = scipy.fftpack.dct(energy_comp, axis=1, type=dct_type, norm=norm)
    mfcc = mfcc[:, 1:n_mfcc+1]
    dmfcc = np.diff(mfcc, axis=0, prepend=mfcc[0, None])
    ddmfcc = np.diff(dmfcc, axis=0, prepend=dmfcc[0, None])
    return np.hstack((mfcc, dmfcc, ddmfcc))


def cubicmfcc(x, n_mfcc=13, dct_type=2, norm='ortho', filtered=False,
              filt_kwargs=None, framed=False, frame_kwargs=None,
              energy=None):
    """
    Mel-frequency cepstral coefficients.

    Calculates the mel-frequency cepstral coefficients (MFCC). DC term is not
    returned. Deltas and double deltas are also returned. Uses cubic
    compression.

    Parameters
    ----------
    x : array_like
        Input signal.
    n_mfcc : int, optional
        Number of MFCCs to return, DC term not included. Default is 13.
    dct_type : {1, 2, 3, 4}, optional
        Type of the discrete cosine transform (DCT). See the `np.fft.dct`
        documentation. Default is 2.
    norm : {'backward', 'ortho', 'forward'}, optional
        Normalization mode of the discrete cosine transform (DCT). See the
        `np.fft.dct` documentation. Default is `'ortho'`.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.
    energy : array_like
        Pre-computed filterbank energies. Shape `(n_samples, n_filters)`.
        Default is None.

    Returns
    -------
    mfcc : array_like
        Mel-frequency cepstral coefficients. Shape `(n_frames, n_mfcc*3)`.
    """
    if energy is None:
        x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
        x = x.mean(axis=-1)  # average channels
        energy = x**2  # get energy
        energy = energy.mean(axis=1)  # average each frame
    energy_comp = energy**(1/3)
    mfcc = scipy.fftpack.dct(energy_comp, axis=1, type=dct_type, norm=norm)
    mfcc = mfcc[:, 1:n_mfcc+1]
    dmfcc = np.diff(mfcc, axis=0, prepend=mfcc[0, None])
    ddmfcc = np.diff(dmfcc, axis=0, prepend=dmfcc[0, None])
    return np.hstack((mfcc, dmfcc, ddmfcc))


def pdf(x, filtered=False, filt_kwargs=None, framed=False, frame_kwargs=None,
        log=False, energy=None):
    """
    Probability density function estimate.

    Calculates the probability density function estimate (PDF) in each
    time-frequency unit.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.
    log : bool, optional
        If `True`, logarithmic compression is applied to the output. Default is
        `False`.
    energy : array_like
        Pre-computed filterbank energies. Shape `(n_samples, n_filters)`.
        Default is None.

    Returns
    -------
    pdf : array_like
        Probability density function estimate. Shape `(n_frames, n_filters)`.
    """
    if energy is None:
        x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
        x = x.mean(axis=-1)  # average channels
        energy = x**2  # get energy
        energy = energy.mean(axis=1)  # average each frame
    total_energy = energy.sum(axis=1, keepdims=True)  # energy across bands
    pdf = energy/(total_energy + np.nextafter(0, 1))  # normalization
    if log:
        pdf = np.log(pdf + np.nextafter(0, 1))
    return pdf


def pdfcc(x, n_pdfcc=13, dct_type=2, norm='ortho', filtered=False,
          filt_kwargs=None, framed=False, frame_kwargs=None,
          energy=None):
    """
    DCT-compressed PDF feature

    Calculates a DCT-compressed version the PDF feature. It is calculated
    exactly like the MFCC, but the energy is normalized before applying the
    logarithmic and DCT transforms such that it lies between 0 and 1. Deltas
    and double deltas are also returned.

    Parameters
    ----------
    x : array_like
        Input signal.
    n_pffcc : int, optional
        Number of coefficients to return, DC term not included. Default is 13.
    dct_type : {1, 2, 3, 4}, optional
        Type of the discrete cosine transform (DCT). See the `np.fft.dct`
        documentation. Default is 2.
    norm : {'backward', 'ortho', 'forward'}, optional
        Normalization mode of the discrete cosine transform (DCT). See the
        `np.fft.dct` documentation. Default is `'ortho'`.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.
    energy : array_like
        Pre-computed filterbank energies. Shape `(n_samples, n_filters)`.
        Default is None.

    Returns
    -------
    pdfcc : array_like
        DCT-compressed PDF feature. Shape `(n_frames, n_pdfcc*3)`.
    """
    if energy is None:
        x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
        x = x.mean(axis=-1)  # average channels
        energy = x**2  # get energy
        energy = energy.mean(axis=1)  # average each frame
    total_energy = energy.sum(axis=1, keepdims=True)  # energy across bands
    energy = energy/(total_energy + np.nextafter(0, 1))  # normalization
    log_energy = np.log(energy + np.nextafter(0, 1))
    pdfcc = scipy.fftpack.dct(log_energy, axis=1, type=dct_type, norm=norm)
    pdfcc = pdfcc[:, 1:n_pdfcc+1]
    dpdfcc = np.diff(pdfcc, axis=0, prepend=pdfcc[0, None])
    ddpdfcc = np.diff(dpdfcc, axis=0, prepend=dpdfcc[0, None])
    return np.hstack((pdfcc, dpdfcc, ddpdfcc))


def logpdf(x, filtered=False, filt_kwargs=None, framed=False,
           frame_kwargs=None, energy=None):
    """
    Log-compressed probability density function estimate.

    Calculates the log-compressed probability density function estimate in each
    time-frequency unit.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.
    energy : array_like
        Pre-computed filterbank energies. Shape `(n_samples, n_filters)`.
        Default is None.

    Returns
    -------
    logpdf : array_like
        Log-compressed probability density function estimate. Shape
        `(n_frames, n_filters)`.
    """
    return pdf(x, filtered=filtered, filt_kwargs=filt_kwargs, framed=framed,
               frame_kwargs=frame_kwargs, log=True, energy=energy)


def fbe(x, filtered=False, filt_kwargs=None, framed=False, frame_kwargs=None,
        log=False, energy=None):
    """
    Filterbank energies.

    Calculates the energy in each time-frequency unit.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.
    log : bool, optional
        If `True`, logarithmic compression is applied to the output. Default is
        `False`.
    energy : array_like
        Pre-computed filterbank energies. Shape `(n_samples, n_filters)`.
        Default is None.

    Returns
    -------
    fbe : array_like
        Filterbank energies. Shape `(n_frames, n_filters)`.
    """
    if energy is None:
        x = _check_input(x, filtered, filt_kwargs, framed, frame_kwargs)
        x = x.mean(axis=-1)  # average channels
        energy = x**2  # get energy
        energy = energy.mean(axis=1)  # average each frame
    if log:
        energy = np.log(energy + np.nextafter(0, 1))
    return energy


def logfbe(x, filtered=False, filt_kwargs=None, framed=False,
           frame_kwargs=None, energy=None):
    """
    Log-compressed filterbank energies.

    Calculates a log-compressed version of the energy in each time-frequency
    unit.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.
    energy : array_like
        Pre-computed filterbank energies. Shape `(n_samples, n_filters)`.
        Default is None.

    Returns
    -------
    logfbe : array_like
        Log-compressed filterbank energies. Shape `(n_frames, n_filters)`.
    """
    return fbe(x, filtered=filtered, filt_kwargs=filt_kwargs, framed=framed,
               frame_kwargs=frame_kwargs, log=True, energy=energy)


def _check_input(x, filtered=False, filt_kwargs=None, framed=False,
                 frame_kwargs=None):
    """
    Input check prior to feature calculation.

    Checks input shape before feature calculation and transforms it if
    necesarry. Transformations are namely filtering and framing.

    Parameters
    ----------
    x : array_like
        Input signal.
    filtered : bool, optional
        If `True`, the input signal `x` is assumed to be already filtered. It
        should then have shape `(n_samples, n_filters, 2)`. Else, the input is
        filtered using `~filters.filt` before calculation and should have
        shape `(n_samples, 2)`. Default is False.
    filt_kwargs : dict or None, optional
        Keyword arguments passed to `~filters.filt`. Used only if `filtered`
        is `False`. Default is `None`, which means no keyword arguments are
        passed.
    framed : bool, optional
        If `True`, the input signal `x` is assumed to be already framed. It
        should then have shape `(n_frames, frame_length, n_filters, 2)`. Else,
        the input is framed before calculation and should have shape
        `(n_samples, n_filters, 2)`. Default is False.
    frame_kwargs : dict or None, optional
        Keyword arguments passed to `~utils.frame`. Used only if `framed` is
        `False`. Default is `None`, which means no keyword arguments are
        passed.

    Returns
    -------
    y :
        Transformed input.
    """
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
