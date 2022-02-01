import numpy as np
import scipy.signal


def freq_to_erb(f):
    """
    Frequency-to-ERB scale.

    Converts frequency values in hertz to ERB (equivalent rectangular
    bandwidth) values.

    Parameters
    ----------
    f : array-like
        Frequency values in hertz.

    Returns
    -------
    erb : array-like
        ERB values.
    """
    f = np.asarray(f)
    erb = 21.4*np.log10(1 + 0.00437*f)
    return erb


def erb_to_freq(erb):
    """
    ERB-to-frequency scale.

    Converts ERB (equivalent rectangular bandwidth) values to frequency
    values in hertz.

    Parameters
    ----------
    erb : array-like
        ERB values.

    Returns
    -------
    f : array-like
        Frequency values in hertz.
    """
    erb = np.asarray(erb)
    f = (10**(erb/21.4) - 1)/0.00437
    return f


def freq_to_mel(f):
    """
    Hertz-to-mel scale.

    Converts frequency values in hertz to mel values.

    Parameters
    ----------
    f : array-like
        Frequency values in hertz.

    Returns
    -------
    mel : array-like
        Mel values.
    """
    f = np.asarray(f)
    mel = 2595*np.log10(1 + f/700)
    return mel


def mel_to_freq(mel):
    """
    Mel-to-hertz scale.

    Converts mel values to frequency values in hertz.

    Parameters
    ----------
    mel : array-like
        Mel values.

    Returns
    -------
    f : array-like
        Frequency values in hertz.
    """
    mel = np.asarray(mel)
    f = 700*(10**(mel/2595) - 1)
    return f


def gammatone_iir(fc, fs=16e3, order=4):
    """
    Gammatone filter.

    Coefficients for a digital IIR gammatone filter. Inspired from the AMT
    Matlab/Octave toolbox, and from the lyon1996all and katsiamis2007practical
    papers.

    Parameters
    ----------
    fc : float or int
        Center frequency of the filter.
    fs : float or int, optional
        Sampling frequency. Default is 16e3.
    order: int, optional
        Fitler order. Default is 16e3.

    Returns
    -------
    b : array_like
        Numerator coefficients. Length `order+1`.
    a : array_like
        Denominator coefficients. Length `2*order+1`.
    """
    fc = np.asarray(fc)
    ERB = 24.7*(1 + 4.37*fc*1e-3)
    beta = 1.019*ERB
    pole = np.exp(-2*np.pi*(1j*fc+beta)/fs)
    zero = np.real(pole)
    a = np.poly(np.hstack((pole*np.ones(order), np.conj(pole)*np.ones(order))))
    b = np.poly(zero*np.ones(order))
    ejwc = np.exp(1j*2*np.pi*fc/fs)
    gain = np.abs((ejwc - zero)/((ejwc - pole)*(ejwc - np.conj(pole))))**order
    return b/gain, a


def gammatone_filterbank(n_filters=64, f_min=50, f_max=8000, fs=16e3,
                         order=4):
    """
    Gammatone filterbank.

    Coefficients for a bank of gammatone IIR filters equally spaced on an
    ERB-rate scale.

    Parameters
    ----------
    n_filters : int, optional
        Number of filters. Default is 64.
    f_min : int or float, optional
        Center frequency of the lowest filter. Default is 50.
    f_max : int or float, optional
        Center frequency of the highest filter. Default is 8000.
    fs : int or float, optional
        Sampling frequency. Default is 16e3.

    Returns
    -------
    b : array_like
        Numerator coefficients. Shape `(n_filters, order+1)`.
    a : array_like
        Denominator coefficients. Shape `(n_filters, 2*order+1)`.
    fc : array_like
        Center frequencies. Length `n_filters`.
    """
    erb_min, erb_max = freq_to_erb([f_min, f_max])
    erb = np.linspace(erb_min, erb_max, n_filters)
    fc = erb_to_freq(erb)
    b = np.zeros((n_filters, order+1))
    a = np.zeros((n_filters, 2*order+1))
    for i in range(n_filters):
        b[i], a[i] = gammatone_iir(fc[i], fs, order)
    return b, a, fc


def mel_iir(f_low, f_high, fs=16e3, order=4):
    """
    Mel filter.

    Coefficients for a IIR mel filter. The filter is actually a bandpass
    Butterworth filter.

    Parameters
    ----------
    f_low : int or float
        Lower bandwidth frequency.
    f_high : int or float
        Upper bandwidth frequency.
    fs : int or float, optional
        Sampling frequency. Default is 16e3.
    order : int, optional
        Filter order. Default is 4.

    Returns
    -------
    b : array_like
        Numerator coefficients. Length `order+1`.
    a : array_like
        Denominator coefficients. Length `order+1`.
    """
    if order % 2 != 0:
        raise ValueError('order must be even')
    return scipy.signal.butter(order//2, [f_low, f_high], 'bandpass', fs=fs)


def mel_filterbank(n_filters=64, f_min=50, f_max=8000, fs=16e3, order=4):
    """
    Mel filterbank.

    Coefficients for a bank of Butterworth filters equally spaced on a mel
    scale.

    Parameters
    ----------
    n_filters : int, optional
        Number of filters. Default is 64.
    f_min : int or float, optional
        Lower bandwidth frequency of the lowest filter. Default is 50.
    f_max : int or float, optional
        Upper bandwidth frequency of the highest filter. Default is 8000.
    fs : int or float, optional
        Sampling frequency. Default is 16e3.
    order : int, optional
        Order of the filters. Default is 4.

    Returns
    -------
    b : array_like
        Numerator coefficients. Shape `(n_filters, order+1)`.
    a : array_like
        Denominator coefficients. Shape `(n_filters, order+1)`.
    fc :
        Center frequencies.
    """
    mel_min, mel_max = freq_to_mel([f_min, f_max])
    mel = np.linspace(mel_min, mel_max, n_filters+2)
    f_all = mel_to_freq(mel)
    fc = f_all[1:-1]
    f_low = np.sqrt(f_all[:-2]*fc)
    f_high = np.sqrt(fc*f_all[2:])
    b = np.zeros((n_filters, order+1))
    a = np.zeros((n_filters, order+1))
    for i in range(n_filters):
        b[i], a[i] = mel_iir(f_low[i], f_high[i], fs, order)
    return b, a, fc


class Filterbank:
    """
    Main filterbank object.
    """
    def __init__(self, kind='mel', n_filters=64, f_min=50, f_max=8000,
                 fs=16e3, order=4):
        self.kind = kind
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.fs = fs
        self.order = order
        if kind == 'mel':
            filterbank = mel_filterbank
        elif kind == 'gammatone':
            filterbank = gammatone_filterbank
        else:
            raise ValueError(f'kind must be mel or gammatone, got {kind}')
        self.b, self.a, self.fc = filterbank(
            n_filters=n_filters,
            f_min=f_min,
            f_max=f_max,
            fs=fs,
            order=order,
        )

    def __call__(self, x, axis=-1):
        return self.filt(x, axis=axis)

    def filt(self, x, axis=-1):
        output = []
        for i in range(self.n_filters):
            output.append(
                scipy.signal.lfilter(
                    self.b[i], self.a[i], x, axis=axis
                )
            )
        return np.stack(output)

    def rfilt(self, x, axis=-1):
        if x.shape[0] != self.n_filters:
            raise ValueError('input size along first dimension '
                             f'({x.shape[0]}) does not match the number of '
                             f'filters ({self.n_filters})')
        output = []
        for i in range(self.n_filters):
            output.append(
                scipy.signal.lfilter(
                    self.b[i], self.a[i], np.flip(x[i], axis), axis=axis
                )
            )
        output = np.sum(output, axis=0)
        output = np.flip(output, axis=axis)
        return output
