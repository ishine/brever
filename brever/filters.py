import numpy as np
import scipy.signal

from .utils import freq_to_erb, erb_to_freq, freq_to_mel, mel_to_freq
from .utils import fft_freqs


def gammatone_iir(fc, fs=16e3):
    '''
    Coefficients for a digital IIR gammatone filter of order 4. Inspired from
    the AMT Matlab/Octave toolbox, and from the lyon1996all and
    katsiamis2007practical papers.

    Parameters:
        fc:
            Center frequency of the filter.
        fs:
            Sampling frequency in hertz.

    Returns:
        b:
            Numerator coefficients.
        a:
            Denominator coefficients.
    '''
    fc = np.asarray(fc)
    ERB = 24.7*(1 + 4.37*fc*1e-3)
    beta = 1.019*ERB
    order = 4
    pole = np.exp(-2*np.pi*(1j*fc+beta)/fs)
    zero = np.real(pole)
    a = np.poly(np.hstack((pole*np.ones(order), np.conj(pole)*np.ones(order))))
    b = np.poly(zero*np.ones(order))
    ejwc = np.exp(1j*2*np.pi*fc/fs)
    gain = np.abs((ejwc - zero)/((ejwc - pole)*(ejwc - np.conj(pole))))**order
    return b/gain, a


def gammatone_iir_filterbank(n_filters=64, f_min=50, f_max=8000, fs=16e3):
    '''
    Coefficients for a bank of 4-th order gammatone IIR filters equally spaced
    on an ERB-rate scale.

    Parameters:
        n_filters:
            Number of filters.
        f_min:
            Center frequency of the lowest filter.
        f_max:
            Center frequency of the highest filter.
        fs:
            Sampling frequency.

    Returns:
        b:
            Numerator coefficients. Size n_filters*9.
        a:
            Denominator coefficients. Size n_filters*5.
        fc:
            Center frequencies.
    '''
    erb_min, erb_max = freq_to_erb([f_min, f_max])
    erb = np.linspace(erb_min, erb_max, n_filters)
    fc = erb_to_freq(erb)
    b = np.zeros((n_filters, 5))
    a = np.zeros((n_filters, 9))
    for i in range(n_filters):
        b[i], a[i] = gammatone_iir(fc[i], fs)
    return b, a, fc


def gammatone_filt(x, n_filters=64, f_min=50, f_max=8000, fs=16e3):
    '''
    Filter a signal through a bank of gammatone filters equally spaced on an
    ERB-rate scale.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional
            must have shape n_samples*n_channels.
        n_filters:
            Number of filters.
        f_min:
            Minimum center frequency in hertz.
        f_max:
            Maximum center frequency in hertz.
        fs:
            Sampling frequency in hertz.

    Returns:
        x_filt:
            Decomposed signal. Shape n_samples*n_filters, or
            n_samples*n_filters*n_channels if multichannel input.
        fc:
            Center frequencies.
    '''
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_samples, n_channels = x.shape
    x_filt = np.zeros((n_samples, n_filters, n_channels))
    b, a, fc = gammatone_iir_filterbank(n_filters, f_min, f_max, fs)
    for i in range(n_filters):
        x_filt[:, i, :] = scipy.signal.lfilter(b[i], a[i], x, axis=0)
    return x_filt.squeeze(), fc


def mel_iir(f_low, f_high, fs=16e3, order=6):
    '''
    Coefficients for a bandpass Butterworth filter.

    Parameters:
        f_low:
            Lower bandwidth frequency.
        f_high:
            Upper bandwidth frequency.
        fs:
            Sampling frequency.
        order:
            Filter order.

    Returns:
        b:
            Numerator coefficients.
        a:
            Denominator coefficients.
    '''
    if order % 2 != 0:
        raise ValueError('order must be even')
    b, a = scipy.signal.butter(order//2, [f_low, f_high], 'bandpass', fs=fs)
    return b, a


def mel_iir_filterbank(n_filters=64, f_min=50, f_max=8000, fs=16e3, order=6):
    '''
    Coefficients for a filterbank of Butterworth filters equally spaced on a
    mel scale.

    Parameters:
        n_filters:
            Number of filters.
        f_min:
            Lower frequency of the first filter.
        f_max:
            Higher frequency of the last filter.
        fs:
            Sampling frequency.
        order:
            Order of the filters.

    Returns:
        b:
            Numerator coefficients. Size n_filters*(order+1).
        a:
            Denominator coefficients. Size n_filters*(order+1).
        fc:
            Center frequencies.
    '''
    mel_min, mel_max = freq_to_mel([f_min, f_max])
    mel = np.linspace(mel_min, mel_max, n_filters+2)
    f_all = mel_to_freq(mel)
    fc = f_all[1:-1]
    f_low = np.sqrt(f_all[:-2]*fc)
    f_high = np.sqrt(fc*f_all[2:])
    b, a = np.zeros((2, n_filters, order+1))
    for i in range(n_filters):
        b[i], a[i] = mel_iir(f_low[i], f_high[i], fs, order)
    return b, a, fc


def mel_filt(x, n_filters=64, f_min=50, f_max=8000, fs=16e3, order=6):
    '''
    Filter a signal through a bank of Butterworth filters equally spaced on an
    mel scale.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional
            must have shape n_samples*n_channels.
        n_filters:
            Number of filters.
        f_min:
            Lower frequency of the first filter.
        f_max:
            Higher frequency of the last filter.
        fs:
            Sampling frequency.
        order:
            Order of the filters.

    Returns:
        x_filt:
            Decomposed signal. Shape n_samples*n_filters, or
            n_samples*n_filters*n_channels if multichannel input.
        fc:
            Center frequencies.
    '''
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_samples, n_channels = x.shape
    x_filt = np.zeros((n_samples, n_filters, n_channels))
    b, a, fc = mel_iir_filterbank(n_filters, f_min, f_max, fs, order)
    for i in range(n_filters):
        x_filt[:, i, :] = scipy.signal.lfilter(b[i], a[i], x, axis=0)
    return x_filt.squeeze(), fc


def mel_triangle_filterbank(n_filters=64, n_fft=512, f_min=50, f_max=8000,
                            fs=16e3):
    '''
    Triangular mel filters equally spaced on a mel scale. The output is a
    two-dimensional array with size n_bins*n_filters so that a melspectrogram
    is obtained by multiplying it with a spectrogram.

    Parameters:
        n_filters:
            Number of filters.
        n_fft:
            Number of FFT points.
        f_min:
            Lower frequency of the lowest filter.
        f_max:
            Higher frequency of the highest filter.
        fs:
            Sampling frequency in hertz.

    Returns:
        FB:
            Mel filterbank, with size n_bins*n_filters.
        fc:
            Center frequencies.
    '''
    mel_min, mel_max = freq_to_mel([f_min, f_max])
    mel = np.linspace(mel_min, mel_max, n_filters+2)
    fc = mel_to_freq(mel)
    f = fft_freqs(fs, n_fft, onesided=True)
    FB = np.zeros((len(f), n_filters))
    for i in range(n_filters):
        mask = (fc[i] < f) & (f <= fc[i+1])
        FB[mask, i] = (f[mask]-fc[i])/(fc[i+1]-fc[i])
        mask = (fc[i+1] < f) & (f < fc[i+2])
        FB[mask, i] = (fc[i+2]-f[mask])/(fc[i+2]-fc[i+1])
    return FB, fc[1:-1]
