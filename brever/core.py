import numpy as np
import scipy.signal

from .utils import freq_to_erb, erb_to_freq, freq_to_mel, mel_to_freq
from .utils import fft_freqs, frame


def stft(x, n_fft=512, hop_length=256, frame_length=None, window='hann',
         onesided=True, center=False, normalization=False):
    '''
    STFT computation.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional,
            should have shape n_samples*n_channels.
        n_fft:
            Number of FFT points.
        hop_length:
            Frame shift in samples.
        frame_length:
            Frame length in samples. If None, matches n_fft. If specified and
            higher than n_fft, the frames are zero-padded. If smaller than
            n_fft, the frames are cropped.
        window:
            Window type. Can be a string, an array or a function.
            - If a string, it is passed to scipy.signal.get_window together
            with frame_length. Note that this creates a periodic (asymmetric)
            window, which is recommended in spectral analysis.
            - If an array, it should be one-dimensional with length
            frame_length.
            - If a function, it needs to take frame_length as an argument and
            return an array of length frame_length.
        onesided:
            If True, the one-sided FFT for real signals is computed.
        center:
            If True, the first frame is centered at the first sample by
            zero-padding at the beginning of x, such that the frame of index i
            is centered at i*hop_length.
        normalization:
            If True, the output is divided by n_fft**0.5. This ensures
            Parseval's theorem, i.e. the energy (sum of squares) in each frame
            is the same in both the time domain and the frequency domain.

    Returns:
        X:
            STFT of x with size n_frames*n_fft*n_channels.
    '''
    if frame_length is None:
        frame_length = n_fft
    if callable(window):
        window = window(frame_length)
    elif isinstance(window, str):
        window = scipy.signal.get_window(window, frame_length)
    window = window.reshape(1, -1, 1)
    frames = frame(x, frame_length, hop_length, center)*window
    if onesided:
        X = np.fft.rfft(frames, n_fft, axis=1)
    else:
        X = np.fft.fft(frames, n_fft, axis=1)
    if normalization:
        X /= n_fft**0.5
    return X


def spectrogram(x, n_fft=512, hop_length=256, frame_length=None, window='hann',
                onesided=True, center=False, normalization=False,
                domain='power'):
    '''
    Spectrogram of input signal.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional,
            should have shape n_samples*n_channels.
        n_fft:
            Number of FFT points.
        hop_length:
            Frame shift in samples.
        frame_length:
            Frame length in samples. If None, matches n_fft. If specified and
            higher than n_fft, the frames are zero-padded. If smaller than
            n_fft, the frames are cropped.
        window:
            Window type. Can be a string, an array or a function.
            - If a string, it is passed to scipy.signal.get_window together
            with frame_length. Note that this creates a periodic (asymmetric)
            window, which is recommended in spectral analysis.
            - If an array, it should be one-dimensional with length
            frame_length.
            - If a function, it needs to take frame_length as an argument and
            return an array of length frame_length.
        onesided:
            If True, the one-sided FFT for real signals is computed.
        center:
            If True, the first frame is centered at the first sample by
            zero-padding at the beginning of x, such that the frame of index i
            is centered at i*hop_length.
        normalization:
            If True, the output is divided by n_fft**0.5. This ensures
            Parseval's theorem, i.e. the energy (sum of squares) in each frame
            is the same in both the time domain and the frequency domain.
        domain:
            Output domain. Can be either 'mag', 'power' or 'dB'.

    Returns:
        X:
            STFT of x with size n_frames*n_fft*n_channels.
    '''
    X = stft(x, n_fft, hop_length, frame_length, window, onesided, center,
             normalization)
    if domain == 'mag':
        X = np.abs(X)
    elif domain == 'power':
        X = np.abs(X)**2
    elif domain == 'dB':
        X = 20*np.log10(np.abs(X) + 1e-6)
    else:
        raise ValueError('domain should be either mag, power or dB')
    return X


def gammatone_coef(fc, fs=16e3):
    '''
    Get coefficients for a digital IIR gammatone filter. Inspired from the
    AMT toolbox.

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


def gammatone_filt(x, n_filters=64, f_min=50, f_max=8000, fs=16e3):
    '''
    Filter a signal through a bank of gammatone filters equally spaced on an
    ERB-rate scale.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional,
            should have shape n_samples*n_channels.
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
            Decomposed signal. Shape n_samples*n_filters*n_channels.
        fc:
            Center frequencies.
    '''
    erb_min, erb_max = freq_to_erb([f_min, f_max])
    erb = np.linspace(erb_min, erb_max, n_filters)
    fc = erb_to_freq(erb)
    n_samples, n_channels = x.shape
    x_filt = np.zeros((n_samples, n_filters, n_channels))
    for i in range(n_filters):
        b, a = gammatone_coef(fc[i], fs)
        x_filt[:, i, :] = scipy.signal.lfilter(b, a, x, axis=0)
    return x_filt, fc


def cochleagram(x, n_filters=64, f_min=50, f_max=8000, fs=16e3, rectify=True,
                compression='square'):
    '''
    Cochleagram analysis using a gammatone impulse response filterbank.

    Parameters:
        x:
            Input signal.
        n_filters:
            Number of filters.
        f_min:
            Minimum center frequency in hertz.
        f_max:
            Maximum center frequency in hertz.
        fs:
            Sampling frequency in hertz.
        rectify:
            If True, the output of each filter is half wave rectified.
        compression:
            Can be either 'square', 'cube', 'log' or 'none'. Used only if
            rectify is True.

    Returns:
        C:
            Cochleagram. Size n_filters*len(x).
        fc:
            Center frequencies in hertz.
    '''
    if compression not in ['square', 'cube', 'log', 'none']:
        raise ValueError(('compression should be either square, cube, log or '
                          'none'))
    C, fc = gammatone_filt(x, n_filters, f_min, f_max, fs)
    if rectify:
        C = np.maximum(0, C)
        if compression == 'square':
            C = C**0.5
        elif compression == 'cube':
            C = C**(1/3)
        elif compression == 'log':
            C = np.log(C + 1e-6)
    return C, fc


def mel_filterbank(n_filters=64, n_fft=512, f_min=50, f_max=8000, fs=16e3):
    '''
    Mel filters equally spaced on a mel scale. The output is a two-dimensional
    array with size (n_fft//2+1)*n_filters so that a melspectrogram is
    obtained by multiplying it with a spectrogram.

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
            Mel filterbank, with size (n_fft//2+1)*n_filters.
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
