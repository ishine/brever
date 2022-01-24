from functools import partial
import queue
import threading

import numpy as np
import scipy.signal

from .utils import (fft_freqs, freq_to_erb, erb_to_freq, freq_to_mel,
                    mel_to_freq, frame)


def gammatone_iir(fc, fs=16e3, order=4):
    '''
    Coefficients for a digital IIR gammatone filter. Inspired from the AMT
    Matlab/Octave toolbox, and from the lyon1996all and katsiamis2007practical
    papers.

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
    pole = np.exp(-2*np.pi*(1j*fc+beta)/fs)
    zero = np.real(pole)
    a = np.poly(np.hstack((pole*np.ones(order), np.conj(pole)*np.ones(order))))
    b = np.poly(zero*np.ones(order))
    ejwc = np.exp(1j*2*np.pi*fc/fs)
    gain = np.abs((ejwc - zero)/((ejwc - pole)*(ejwc - np.conj(pole))))**order
    return b/gain, a


def gammatone_filterbank(n_filters=64, f_min=50, f_max=8000, fs=16e3,
                         order=4):
    '''
    Coefficients for a bank of gammatone IIR filters equally spaced on an
    ERB-rate scale.

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
        b[i], a[i] = gammatone_iir(fc[i], fs, order)
    return b, a, fc


def gammatone_filt(x, n_filters=64, f_min=50, f_max=8000, fs=16e3, order=4):
    '''
    Filter a signal through a bank of gammatone IIR filters equally spaced on
    an ERB-rate scale.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional
            must have shape n_samples*n_channels.
        n_filters:
            Number of filters.
        f_min:
            Center frequency of the lowest filter.
        f_max:
            Center frequency of the highest filter.
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
    return filt(x, 'gammatone', n_filters, f_min, f_max, fs, order)


def mel_iir(f_low, f_high, fs=16e3, order=4, output='ba'):
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
        output:
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or
            second-order sections ('sos').

    Returns:
        b, a:
            Numerator and denominator coefficients of the IIR filter. Only
            returned if `output='ba'`.
        z, p, k:
            Zeros, poles, and system gain of the IIR filter transfer function.
            Only returned if `output='zpk'`.
        sos:
            Second-order sections representation of the IIR filter. Only
            returned if `output=='sos'`.
    '''
    if order % 2 != 0:
        raise ValueError('order must be even')
    return scipy.signal.butter(order//2, [f_low, f_high], 'bandpass', fs=fs,
                               output=output)


def mel_filterbank(n_filters=64, f_min=50, f_max=8000, fs=16e3, order=4,
                   output='ba'):
    '''
    Coefficients for a bank of Butterworth filters equally spaced on a mel
    scale.

    Parameters:
        n_filters:
            Number of filters.
        f_min:
            Lower bandwidth frequency of the lowest filter.
        f_max:
            Upper bandwidth frequency of the highest filter.
        fs:
            Sampling frequency.
        order:
            Order of the filters.
        output:
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or
            second-order sections ('sos').

    Returns:
        b, a:
            Numerator and denominator coefficients of the IIR filter. Only
            returned if `output='ba'`.
        z, p, k:
            Zeros, poles, and system gain of the IIR filter transfer function.
            Only returned if `output='zpk'`.
        sos:
            Second-order sections representation of the IIR filter. Only
            returned if `output=='sos'`.
        fc:
            Center frequencies.
    '''
    mel_min, mel_max = freq_to_mel([f_min, f_max])
    mel = np.linspace(mel_min, mel_max, n_filters+2)
    f_all = mel_to_freq(mel)
    fc = f_all[1:-1]
    f_low = np.sqrt(f_all[:-2]*fc)
    f_high = np.sqrt(fc*f_all[2:])
    filters = []
    for i in range(n_filters):
        filters.append(mel_iir(f_low[i], f_high[i], fs, order, output))
    return filters, fc


def filt(x, filter_type='mel', n_filters=64, f_min=50, f_max=8000,
         fs=16e3, order=4):
    '''
    Filter a signal through a bank of IIR filters or the selected type.

    Parameters:
        x:
            Input array. Can be one- or two-dimensional. If two-dimensional
            must have shape n_samples*n_channels.
        filter_type:
            Filter type. Currently either 'gammatone' or 'mel'.
        n_filters:
            Number of filters.
        f_min:
            Lower frequency of the filterbank.
        f_max:
            Higher frequency of the filterbank.
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
    if filter_type == 'mel':
        b, a, fc = mel_filterbank(n_filters, f_min, f_max, fs, order)
    elif filter_type == 'gammatone':
        b, a, fc = gammatone_filterbank(n_filters, f_min, f_max, fs, order)
    else:
        raise ValueError('filter_type must be mel or gammatone')
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
    f = fft_freqs(fs, n_fft)
    FB = np.zeros((len(f), n_filters))
    for i in range(n_filters):
        mask = (fc[i] < f) & (f <= fc[i+1])
        FB[mask, i] = (f[mask]-fc[i])/(fc[i+1]-fc[i])
        mask = (fc[i+1] < f) & (f < fc[i+2])
        FB[mask, i] = (fc[i+2]-f[mask])/(fc[i+2]-fc[i+1])
    return FB, fc[1:-1]


def stft(x, n_fft=512, hop_length=256, frame_length=None, window='hann',
         onesided=True, center=False):
    '''
    Short-time Fourier transform.

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

    Returns:
        X:
            STFT of x with size n_frames*n_bins, or n_frames*n_bins*n_channels
            if multichannel input.
    '''
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if frame_length is None:
        frame_length = n_fft
    frames = frame(x, frame_length, hop_length, window, center)
    if onesided:
        X = np.fft.rfft(frames, n_fft, axis=1)
    else:
        X = np.fft.fft(frames, n_fft, axis=1)
    return X.squeeze()


def istft(X, frame_length=512, hop_length=256, window='hann', onesided=True,
          center=False, trim=None):
    '''
    Inverse short-time Fourier transform.
    '''
    if callable(window):
        window = window(frame_length)
    elif isinstance(window, str):
        window = scipy.signal.get_window(window, frame_length)
    if onesided:
        frames = np.fft.irfft(X, frame_length, axis=1)
    else:
        frames = np.fft.ifft(X, frame_length, axis=1)
    n_frames = len(frames)
    n_samples = (n_frames-1)*hop_length + frame_length
    output_shape = np.zeros(X.ndim - 1, int)
    output_shape[0] = n_samples
    output_shape[1:] = X.shape[2:]
    x = np.zeros(output_shape)
    for i in range(n_frames):
        j = i*hop_length
        x[j:j+frame_length] += window*frames[i]
    if trim is not None:
        x = x[:trim]
    return x


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
    X = stft(x, n_fft=n_fft, hop_length=hop_length, frame_length=frame_length,
             window=window, center=center, normalization=normalization)
    if domain == 'mag':
        S = np.abs(X)
    elif domain == 'power':
        S = np.abs(X)**2
    elif domain == 'dB':
        S = 20*np.log10(np.abs(X) + np.nextafter(0, 1))
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
        M = 20*np.log10(np.abs(M) + np.nextafter(0, 1))
    elif input_domain == 'power' and output_domain == 'mag':
        M = M**0.5
    elif input_domain == 'power' and output_domain == 'dB':
        M = 10*np.log10(np.abs(M) + np.nextafter(0, 1))
    elif input_domain == 'dB' and output_domain == 'mag':
        M = 10**(M/20)
    elif input_domain == 'dB' and output_domain == 'power':
        M = 10**(M/10)
    return M, fc


def cochleagram(x, n_filters=64, f_min=50, f_max=8000, fs=16e3, rectify=True,
                compression='square', order=4):
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
        order:
            Filter order.

    Returns:
        C:
            Cochleagram. Size n_filters*len(x), or n_filters*len(x)*n_channels
            if multichannel input.
        fc:
            Center frequencies in hertz.
    '''
    if compression not in ['square', 'cube', 'log', 'none']:
        raise ValueError('compression should be either square, cube, log or '
                         'none')
    C, fc = filt(x, 'gammatone', n_filters, f_min, f_max, fs, order)
    if rectify:
        C = np.maximum(0, C)
        if compression == 'square':
            C = C**0.5
        elif compression == 'cube':
            C = C**(1/3)
        elif compression == 'log':
            C = np.log10(C + np.nextafter(0, 1))
    return C, fc


class Filterbank:
    def __init__(self, kind, n_filters, f_min, f_max, fs, order, output='ba'):
        if output not in ['ba', 'sos']:
            raise ValueError('only "ba" and "sos" outputs are not supported, '
                             f'got "{output}"')
        if output != 'ba' and kind == 'gammatone':
            raise ValueError('only "ba" output is supported for gammatone '
                             f'filterbank, got "{output}"')
        self.kind = kind
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.fs = fs
        self.order = order
        self.output = output
        if kind == 'mel':
            self.filters, self.fc = mel_filterbank(n_filters, f_min, f_max, fs,
                                                   order, output)
        elif kind == 'gammatone':
            self.filters, self.fc = gammatone_filterbank(n_filters, f_min,
                                                         f_max, fs, order)
        else:
            raise ValueError('filtertype must be "mel" or "gammatone", got '
                             f'"{kind}"')

    def filt(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_samples, n_channels = x.shape
        x_filt = np.zeros((n_samples, self.n_filters, n_channels))
        for i in range(self.n_filters):
            if self.output == 'ba':
                filter_func = partial(scipy.signal.lfilter, self.filters[i][0],
                                      self.filters[i][1], axis=0)
            elif self.output == 'sos':
                filter_func = partial(scipy.signal.sosfilt, self.filters[i],
                                      axis=0)
            x_filt[:, i, :] = filter_func(x)
        return x_filt.squeeze()

    def rfilt(self, x_filt):
        if x_filt.ndim == 2:
            x_filt = x_filt[:, :, np.newaxis]
        x = np.zeros((len(x_filt), x_filt.shape[2]))
        for i in range(self.n_filters):
            if self.output == 'ba':
                filter_func = partial(scipy.signal.lfilter, self.filters[i][0],
                                      self.filters[i][1], axis=0)
            elif self.output == 'sos':
                filter_func = partial(scipy.signal.sosfilt, self.filters[i],
                                      axis=0)
            else:
                raise ValueError('wrong output type')
            x += filter_func(x_filt[::-1, i, :])
        return x[::-1].squeeze()


class MultiThreadFilterbank(Filterbank):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _thread_target(self, q, filter_, x, i):
        if self.output == 'ba':
            q.put((i, scipy.signal.lfilter(filter_[0], filter_[1], x, axis=0)))
        elif self.output == 'sos':
            q.put((i, scipy.signal.sosfilt(filter_, x, axis=0)))
        else:
            raise ValueError('wrong output type')

    def filt(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        q = queue.Queue()

        n_samples, n_channels = x.shape
        for i in range(self.n_filters):
            t = threading.Thread(
                target=self._thread_target,
                args=(
                    q,
                    self.filters[i],
                    x,
                    i,
                ),
            )
            t.daemon = True
            t.start()

        x_filt = np.zeros((n_samples, self.n_filters, n_channels))
        for j in range(self.n_filters):
            i, data = q.get()
            x_filt[:, i, :] = data

        return x_filt.squeeze()

    def rfilt(self, x_filt):
        if x_filt.ndim == 2:
            x_filt = x_filt[:, :, np.newaxis]

        q = queue.Queue()

        for i in range(self.n_filters):
            t = threading.Thread(
                target=self._thread_target,
                args=(
                    q,
                    self.filters[i],
                    x_filt[::-1, i, :],
                    i
                ),
            )
            t.daemon = True
            t.start()

        x = np.zeros((len(x_filt), x_filt.shape[2]))
        for j in range(self.n_filters):
            i, data = q.get()
            x += data

        return x[::-1].squeeze()
