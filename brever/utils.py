import numpy as np
import math
import scipy.signal


def rms(x):
    '''
    Root mean square of input signal per channel.

    Parameters:
        x:
            Input signal. Shape n_samples*n_channels.

    Returns:
        rms:
            Root mean square. Length n_channels.
    '''
    return np.mean(x**2, axis=0)**0.5


def zero_pad(x, pad_length, where='after'):
    '''
    Zero-padding along first axis at the end or the beginning an array.

    Parameters:
        x:
            Input array.
        pad_length:
            Number of zeros to append.
        where:
            If 'after', zeros are padded at the end of the array. If 'before',
            they are padded at the beginning.

    Returns:
        x_pad:
            Padded array. Length len(x) + pad_length.
    '''
    padding = np.zeros((x.ndim, 2), int)
    if where == 'before':
        padding[0][0] = pad_length
    elif where == 'after':
        padding[0][1] = pad_length
    elif where == 'both':
        padding[0][0] = pad_length
        padding[0][1] = pad_length
    else:
        raise ValueError('where should be either before, after or both')
    x_pad = np.pad(x, padding)
    return x_pad


def frame(x, frame_length=512, hop_length=256, window='hann', center=False):
    '''
    Slices an array into overlapping frames along first axis.

    Parameters:
        x:
            Input array. Can be multi-dimensional.
        frame_length:
            Frame length in samples.
        hop_length:
            Frame shift in samples.
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

    Returns:
        frames:
            Sliced version of x with size n_frames*frame_length*x.shape[1:].
    '''
    if center:
        x = zero_pad(x, frame_length//2, 'before')
    n_samples = len(x)
    n_frames = math.ceil(max(0, n_samples-frame_length)/hop_length) + 1
    x = zero_pad(x, (n_frames-1)*hop_length + frame_length - n_samples)
    output_shape = np.zeros(x.ndim + 1, int)
    output_shape[[0, 1]] = n_frames, frame_length
    output_shape[2:] = x.shape[1:]
    frames = np.zeros(output_shape, x.dtype)
    for i in range(n_frames):
        j = i*hop_length
        frames[i] = x[j:j+frame_length]
    if callable(window):
        window = window(frame_length)
    elif isinstance(window, str):
        window = scipy.signal.get_window(window, frame_length)
    window_shape = np.ones(frames.ndim, int)
    window_shape[1] = -1
    window = window.reshape(window_shape)
    return frames*window


def wola(X, frame_length=512, hop_length=256, window='hann', trim=None):
    '''
    Weighted overlap-add (WOLA) method.

    Parameters:
        X:
            Input framed signal. Shape n_frames*n_filters.
        frame_length:
            Frame length in samples.
        hop_length:
            Frame shift in samples.
        window:
            Window type. Can be a string, an array or a function.
            - If a string, it is passed to scipy.signal.get_window together
            with frame_length. Note that this creates a periodic (asymmetric)
            window, which is recommended in spectral analysis.
            - If an array, it should be one-dimensional with length
            frame_length.
            - If a function, it needs to take frame_length as an argument and
            return an array of length frame_length.
        trim:
            If an integer, the output is trimmed such that its length is equal
            to trim.

    Returns:
        x:
            Reconstructed signal
    '''
    if callable(window):
        window = window(frame_length)
    elif isinstance(window, str):
        window = scipy.signal.get_window(window, frame_length)
    n_frames, n_filters = X.shape
    n_samples = frame_length + (n_frames-1)*hop_length
    factor = hop_length/window.sum()
    x = np.zeros((n_samples, n_filters))
    for i in range(n_frames):
        j = i*hop_length
        x[j:j+frame_length, :] += np.outer(window, X[i, :])*factor
    if trim is not None:
        x = x[:trim]
    return x


def standardize(x, axis=0):
    '''
    Standardize an array along given axis i.e. remove mean and divide by
    standard deviation.

    Parameters:
        x:
            Input series.

    Returns:
        x_standard
            Standardized series.
    '''
    x = np.asarray(x)
    means = x.mean(axis=axis)
    stds = x.std(axis=axis)
    means = np.expand_dims(means, axis=axis)
    stds = np.expand_dims(stds, axis=axis)
    x_standard = (x - means)/(stds + 1e-10)
    return x_standard


def pca(X, n_components=None, fve=None):
    '''
    Principal component analysis.

    Parameters:
        X:
            Input data. Size n_samples*n_features.
        n_components:
            Number of principal components to return. By default all components
            are returned.
        fve:
            Fraction of variance explained. Can be provided instead of
            n_components such that the returned components account for at least
            fve of variance explained.

    Returns:
        components:
            Principal components. Size n_features*n_components.
        ve:
            Variance explained by each component. Length n_components.
        means:
            Per-feature empirical mean. Length n_features.
    '''
    if n_components is None and fve is None:
        raise ValueError('either n_components or fve must be specified')
    elif n_components is not None and fve is not None:
        raise ValueError('can\'t specify both n_components and fve')
    elif fve is not None and not 0 <= fve <= 1:
        raise ValueError('when specified, fve must be between 0 and 1')
    means = X.mean(axis=0)
    X_center = X-means
    components, ve, _ = np.linalg.svd(X_center.T@X_center)
    ve /= len(X)-1
    if fve is not None:
        if fve == 1:
            n_components = components.shape[1]
        else:
            n_components = np.argmax(np.cumsum(ve)/np.sum(ve) >= fve) + 1
    components = components[:, :n_components]
    ve = ve[:, :n_components]
    return components, ve, means


def freq_to_erb(f):
    '''
    Conversion from frequency in hertz to ERB-rate.

    Parameters:
        f:
            Frequency in hertz.

    Returns:
        erb:
            ERB-rate.
    '''
    f = np.asarray(f)
    erb = 21.4*np.log10(1 + 0.00437*f)
    return erb


def erb_to_freq(erb):
    '''
    Conversion from ERB-rate to frequency in hertz.

    Parameters:
        erb:
            ERB-rate.

    Returns:
        f:
            Frequency in hertz.
    '''
    erb = np.asarray(erb)
    f = (10**(erb/21.4) - 1)/0.00437
    return f


def freq_to_mel(f):
    '''
    Conversion from frequency in hertz to mel scale.

    Parameters:
        f:
            Frequency in hertz.

    Returns:
        mel:
            Mel scale value.
    '''
    f = np.asarray(f)
    mel = 2595*np.log10(1 + f/700)
    return mel


def mel_to_freq(mel):
    '''
    Conversion from mel scale to frequency in hertz.

    Parameters:
        mel:
            Mel scale value.

    Returns:
        f:
            Frequency in hertz.
    '''
    mel = np.asarray(mel)
    f = 700*(10**(mel/2595) - 1)
    return f


def frames_to_time(frames, fs=16e3, hop_length=256):
    '''
    Calculates the time vector for any framed data.

    Parameters:
        frames:
            Framed data. Only its shape along the second axis is used.
        fs:
            Sampling frequency.
        hop_length:
            Frame shift

    Returns:
        t:
            Time vector.
    '''
    t = np.arange(len(frames))*hop_length/fs
    return t


def fft_freqs(fs=16e3, n_fft=512, onesided=True):
    '''
    Calculates the frequency vector for an FFT output.

    Parameters:
        fs:
            Sampling frequency.
        n_fft:
            Number of FFT points.
        onesided:
            If True, only the positive frequencies are returned.

    Returns:
        freqs:
            FFT frequencies.
    '''
    freqs = np.arange(n_fft)*fs/n_fft
    mask = freqs > fs/2
    if onesided:
        freqs = freqs[~mask]
    else:
        freqs[mask] = freqs[mask] - fs
    return freqs


if __name__ == '__main__':

    '''fft_freqs test'''
    import librosa

    fs = 16e3

    n_fft = 8

    print('even number of points, two-sided:')
    print(librosa.fft_frequencies(fs, n_fft))
    print(fft_freqs(fs, n_fft))
    print('ok\n')

    print('even number of points, one-sided:')
    print(fft_freqs(fs, n_fft, onesided=False))
    print(np.fft.fftfreq(n_fft)*fs)
    print(('np.fft.fftfreq is not perfect, the nyquist frequency is '
           'negative!\n'))

    n_fft = 9

    print('odd number of points, two-sided:')
    print(librosa.fft_frequencies(fs, n_fft).round())
    print(fft_freqs(fs, n_fft).round())
    print(('librosa.fft_frequencies is wrong! the nyquist frequency '
           'shouldn\'t be calculated when n_fft is odd!\n'))

    print('odd number of points, two-sided:')
    print(fft_freqs(fs, n_fft, onesided=False).round())
    print((np.fft.fftfreq(n_fft)*fs).round())
    print('ok\n')
