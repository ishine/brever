import numpy as np
import scipy.signal

from .utils import frame
from .filters import gammatone_filt, mel_triangle_filterbank


def stft(x, n_fft=512, hop_length=256, frame_length=None, window='hann',
         onesided=True, center=False, normalization=False):
    '''
    STFT computation.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional
            must have shape n_samples*n_channels.
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
            STFT of x with size n_frames*n_bins, or n_frames*n_bins*n_channels
            if multichannel input.
    '''
    if x.ndim == 1:
        x = x.reshape(-1, 1)
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
    return X.squeeze()


def spectrogram(x, n_fft=512, hop_length=256, frame_length=None, window='hann',
                center=False, normalization=False, domain='power'):
    '''
    Spectrogram of input signal.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional
            must have shape n_samples*n_channels.
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
        S:
            Spectrogram of x with size n_frames*n_bins, or
            n_frames*n_bins*n_channels if multichannel input.
    '''
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    X = stft(x, n_fft, hop_length=hop_length, frame_length=frame_length,
             window=window, center=center, normalization=normalization)
    if domain == 'mag':
        S = np.abs(X)
    elif domain == 'power':
        S = np.abs(X)**2
    elif domain == 'dB':
        S = 20*np.log10(np.abs(X) + 1e-10)
    else:
        raise ValueError('domain should be either mag, power or dB')
    return S.squeeze()


def melspectrogram(x, n_fft=512, hop_length=256, frame_length=None,
                   window='hann', n_filters=64, f_min=50, f_max=8000, fs=16e3,
                   center=False, normalization=False, input_domain='power',
                   output_domain='power'):
    '''
    Mel-spectrogram of input signal.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional
            must have shape n_samples*n_channels.
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
        n_filters:
            Number of mel filters.
        f_min:
            Lower frequency of the lowest mel filter.
        f_max:
            Higher frequency of the highest mel filter.
        fs:
            Sampling frequency in hertz.
        center:
            If True, the first frame is centered at the first sample by
            zero-padding at the beginning of x, such that the frame of index i
            is centered at i*hop_length.
        normalization:
            If True, the output is divided by n_fft**0.5. This ensures
            Parseval's theorem, i.e. the energy (sum of squares) in each frame
            is the same in both the time domain and the frequency domain.
        input_domain:
            Input domain of the filterbank, before grouping into mel bands. Can
            be either 'mag', 'power' or 'dB'.
        output_domain:
            Output domain on the filterbank. Can be either 'mag', 'power' or
            'dB'.

    Returns:
        M:
            Mel-spectrogram, with size n_frames*n_filters.
        fc:
            Center frequencies.
    '''
    S = spectrogram(x, n_fft=n_fft, hop_length=hop_length,
                    frame_length=frame_length, window=window, center=center,
                    normalization=normalization, domain=input_domain)
    FB, fc = mel_triangle_filterbank(n_filters=n_filters, n_fft=n_fft,
                                     f_min=f_min, f_max=f_max, fs=fs)
    M = np.einsum('ijk,jl->ilk', S, FB)
    if input_domain == output_domain:
        pass
    elif input_domain == 'mag' and output_domain == 'power':
        M = M**2
    elif input_domain == 'mag' and output_domain == 'dB':
        M = 20*np.log10(np.abs(M) + 1e-10)
    elif input_domain == 'power' and output_domain == 'mag':
        M = M**0.5
    elif input_domain == 'power' and output_domain == 'dB':
        M = 10*np.log10(np.abs(M) + 1e-10)
    elif input_domain == 'dB' and output_domain == 'mag':
        M = 10**(M/20)
    elif input_domain == 'dB' and output_domain == 'power':
        M = 10**(M/10)
    return M, fc


def cochleagram(x, n_filters=64, f_min=50, f_max=8000, fs=16e3, rectify=True,
                compression='square'):
    '''
    Cochleagram analysis using a gammatone impulse response filterbank.

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
        rectify:
            If True, the output of each filter is half wave rectified.
        compression:
            Can be either 'square', 'cube', 'log' or 'none'. Used only if
            rectify is True.

    Returns:
        C:
            Cochleagram. Size n_filters*len(x), or n_filters*len(x)*n_channels
            if multichannel input.
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
            C = np.log10(C + 1e-6)
    return C, fc
