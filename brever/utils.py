import numpy as np
import math


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

    Return:
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


def frame(x, frame_length, hop_length, center=False):
    '''
    Slices an array into overlapping frames along first axis.

    Parameters:
        x:
            Input array. Can be multi-dimensional.
        frame_length:
            Frame length in samples.
        hop_length:
            Frame shift in samples.
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
    return frames


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
    x_standard = (x - means)/stds
    return x_standard


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
