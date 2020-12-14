import numpy as np
import math
import scipy.signal


def rms(x, axis=0):
    """
    Root mean square.

    Calculates the root mean square (RMS) of input array along given axis.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which to calculate. Default is 0.

    Returns
    -------
    rms : array_like
        RMS values.
    """
    return np.mean(x**2, axis=axis)**0.5


def pad(x, n, axis=0, where='after'):
    """
    Zero-padding.

    Adds zeros at the end and/or the beginning of an array along given axis.

    Parameters
    ----------
    x : array_like
        Input array.
    n : int
        Number of zeros to append.
    axis : int, optional
        Axis along which to pad. Default is 0.
    where : {'after', 'before', 'both'}, optional
        Where to pad the zeros. Default is `'after'`.

    Returns
    -------
    y : array_like
        Padded array.
    """
    padding = np.zeros((x.ndim, 2), int)
    if where == 'before':
        padding[axis][0] = n
    elif where == 'after':
        padding[axis][1] = n
    elif where == 'both':
        padding[axis][0] = n
        padding[axis][1] = n
    else:
        raise ValueError("`where` must be either 'before', 'after' or 'both'")
    return np.pad(x, padding)


def frame(x, frame_length=512, hop_length=256, window="hann", center=False):
    """
    Array framing.

    Slices an array into overlapping frames along first axis.

    Parameters
    ----------
    x : array_like
        Input array. Can be multi-dimensional.
    frame_length : int, optional
        Frame length in samples. Default is 512.
    hop_length : int, optional
        Frame shift in samples. Default is 256.
    window : str or array-like or callable, optional
        Window type. If a string, it is passed to scipy.signal.get_window
        together with `frame_length`. This creates a periodic (asymmetric)
        window, which is usually desirable in spectral analysis. If an array,
        it must be one-dimensional with length `frame_length` and it will be
        used directly as the window. If a callable, it must take `frame_length`
        as argument and return an array of length `frame_length` that will be
        used as the window. Defaults to a Hann window.
    center : bool, optional
        If `True`, the first frame is centered at the first sample in `x` by
        zero-padding at the beginning of `x`, such that the frame of index `i`
        is centered at `i*hop_length`.

    Returns
    -------
    X : array_like
        Framed array with shape `(n_frames, frame_length, *x.shape[1:])`.
    """
    if center:
        x = pad(x, frame_length//2, 'before')
    n_samples = len(x)
    n_frames = math.ceil(max(0, n_samples-frame_length)/hop_length) + 1
    x = pad(x, (n_frames-1)*hop_length + frame_length - n_samples)
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
    """
    Weighted overlap-add (WOLA).

    Reconstructs a 1-D array from a sequence of overlapping frames using the
    Weighted overlap-add (WOLA) method.

    Parameters
    ----------
    X : array_like
        Input framed array with shape `(n_frames, frame_length)`.
    frame_length : int, optional
        Frame length in samples. Default is 512.
    hop_length : int, optional
        Frame shift in samples. Default is 256.
    window : str or array-like or callable, optional
        Window type. If a string, it is passed to scipy.signal.get_window
        together with `frame_length`. This creates a periodic (asymmetric)
        window, which is usually desirable in spectral analysis. If an array,
        it must be one-dimensional with length `frame_length` and it will be
        used directly as the window. If a callable, it must take `frame_length`
        as argument and return an array of length `frame_length` that will be
        used as the window. Defaults to a Hann window.
    trim : int or None, optional
        If set to an integer, the output is trimmed such that its length is
        exactly equal to `trim`. Default is `None`, which means no trimming
        is performed.

    Returns
    -------
    x : array_like
        Reconstructed 1-D array.
    """
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
    """
    Standardization.

    Standardize an array along given axis, i.e. substract mean and divide by
    standard deviation.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which to standardize. Default is 0.

    Returns
    -------
    y : array_like
        Standardized array.
    """
    x = np.asarray(x)
    means = x.mean(axis=axis)
    stds = x.std(axis=axis)
    means = np.expand_dims(means, axis=axis)
    stds = np.expand_dims(stds, axis=axis)
    x_standard = (x - means)/(stds + 1e-10)
    return x_standard


def pca(X, n_components=None, fve=None):
    """
    Principal component analysis (PCA).

    Performs principal component analysis (PCA) of input array.

    Parameters
    ----------
    X : array_like
        Input array with shape `(n_samples, n_features)`.
    n_components : int or None, optional
        Number of principal components to return. Default is `None`, which
        means all components are returned.
    fve : float or None, optional
        Fraction of variance explained. Can be provided instead of
        `n_components` such that the returned components account for at least
        `fve` of variance explained. Must be between 0 and 1. Default is
        `None`, which means it is ignored and the number of returned components
        follows `n_components`. Specifying both `n_components` and `fve` will
        raise an error.

    Returns
    -------
    components : array_like
        Principal components. Shape `(n_features, n_components)`.
    ve : array_like
        Variance explained by each component. Length `n_components`.
    means : array_like
        Per-feature empirical mean. Length `n_features`.
    """
    if n_components is not None and fve is not None:
        raise ValueError("can't specify both `n_components` and `fve`")
    if fve is not None and not 0 <= fve <= 1:
        raise ValueError("`fve` must be between 0 and 1")
    means = X.mean(axis=0)
    X_center = X-means
    components, ve, _ = np.linalg.svd(X_center.T@X_center)
    ve /= len(X)-1
    if n_components is None:
        n_components = len(components)
    if fve is not None:
        if fve == 1:
            n_components = components.shape[1]
        else:
            n_components = np.argmax(np.cumsum(ve)/np.sum(ve) >= fve) + 1
    components = components[:, :n_components]
    ve = ve[:, :n_components]
    return components, ve, means


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


def frames_to_time(frames, fs=16e3, hop_length=256):
    """
    Time vector from framed data.

    Calculates the time vector from any framed data. The first axis of the
    input array is assumed to be the time axis, i.e. the frame index axis.

    Parameters
    ----------
    frames : array-like
        Framed data with shape `(n_frames, frame_length)`.
    fs : float or int, optional
        Sampling frequency. Default is 16e3.
    hop_length : int, optional
        Frame shift in samples. Default is 256.

    Returns
    -------
    t : array_like
        Time vector.
    """
    t = np.arange(len(frames))*hop_length/fs
    return t


def fft_freqs(fs=16e3, n_fft=512, onesided=True):
    """
    Fast Fourier Transform (FFT) frequencies.

    Calculates the frequency vector corresponding to the output of a Fast
    Fourier Transform (FFT).

    Parameters
    ----------
    fs : float or int, optional
        Sampling frequency. Default is 16e3.
    n_fft : int, optional
        Number of FFT points. Default is 512.
    onesided : bool, optional
        If `False`, both positive and negative frequencies are returned, which
        corresponds to the output from `np.fft.fft`. Default is `False`, which
        means only positive frequencies are returned; this corresponds to the
        output from `np.fft.rfft`.

    Returns
    -------
    freqs : array_like
        Frequency vector.
    """
    freqs = np.arange(n_fft)*fs/n_fft
    mask = freqs > fs/2
    if onesided:
        freqs = freqs[~mask]
    else:
        freqs[mask] = freqs[mask] - fs
    return freqs


def dct(x, n_coef):
    """
    Type-II discrete cosine transform (DCT) compression.

    Performs discrete cosine transform (DCT) compression of 2-D input data
    along first axis. The 0-th order term (DC component) is not returned.

    Parameters
    ----------
    x : array_like
        Input array to compress. Shape `(n_samples, n_features)`.
    n_coef : int
        Number of DCT coefficients.

    Returns
    -------
    y : array_like
        Output DCT-compressed array. Shape `(n_coef, n_features)`.
    """
    N = len(x)
    n = np.arange(N) + 0.5
    k = np.arange(n_coef) + 1  # DC term discarded
    DCT = np.cos(np.pi*np.outer(k, n)/N)
    return DCT@x


def segmental_scores(*args, frame_length=160, hop_length=160, DRdB=45,
                     sdr='default'):
    """
    Segmental speech signal-to-noise ratio (segSSNR) and noise reduction
    (segNR) scores.

    Calculates a segmental speech signal-to-noise ratio (segSSNR) score and as
    many segmental noise reduction (segNR) scores as extra pairs of signals
    provided. A voice activity detection (VAD) is performed on the first
    argument to filter the frames over which the scores are averaged.

    Parameters
    ----------
    *args : array_like
        Input signals. The signals must come as a sequence of alternated
        reference and enhanced signals, e.g. `segmental_scores(x_ref, x_hat,
        y_ref, y_hat, ...)`, and the first two signals must be the reference
        and enhanced target speech signals.
    frame_length : int, optional
        Frame length in samples. Default is 160.
    hop_length : int, optional
        Frame shift in samples. Default is 160, which means there is no overlap
        between frames.
    DRdB : int or float, optional
        Dynamic range in dB relative to the global maximum of 0 dB within which
        the short-term energy of the target signal reflects speech activity.
        This energy-based voice activity detector (VAD) is used to discard
        low-energy frames. Default is 45.
    sdr : {'default', 'si', 'sd'}
        Scaling mode when calculating the signal-to-distorsion (SDR). `'si'`
        means scale-invariant, while `'sd'` means scale-dependent. Default is
        `'default'`, which means the widely used definition of the SDR is used.
        See [1]_ for the definition of each mode.

    .. [1] J. L. Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR –
           Half-baked or Well Done?," ICASSP 2019 - 2019 IEEE International
           Conference on Acoustics, Speech and Signal Processing (ICASSP),
           Brighton, United Kingdom, 2019, pp. 626-630, doi:
           10.1109/ICASSP.2019.8683855.

    Returns
    -------
    scores : list of float
        List of scores. The first element is a segmental speech signal-to-noise
        ratio (segSSNR) score, and the next scores if any are segmental noise
        reduction (segNR) scores. E.g. the output from `segmental_scores(s_ref,
        s_hat, n1_ref, n1_hat, n2_ref, n2_hat)` is a list of 3 elements
        `[segSSNR, segNR1, segNR2]`, where `segSSNR` is a segSSNR score
        calculated from reference speech signal `s_ref` and enhanced speech
        signal `s_hat`, and `segNR1` and `segNR2` are segNR scores calculated
        respectively from reference noise signal `n1_ref` and enhanced noise
        signal `n1_hat`, and reference noise signal `n2_ref` and enhanced noise
        signal `n2_hat`.
    """
    if len(args) % 2 != 0:
        raise ValueError('odd number of arguments; arguments must come as a '
                         'list of alternated reference and enhanced signals, '
                         'e.g. `segmental_scores(x_ref, x_hat, y_ref, y_hat, '
                         '...)`, and the first pair of signals must be the '
                         'reference and enhanced target speech signals')
    if any(arg.ndim > 2 for arg in args):
        raise ValueError('inputs must be at most 2-dimensional')
    if any(arg.ndim == 0 for arg in args):
        raise ValueError('got a 0-dimensional input')
    args = list(args)
    for i, arg in enumerate(args):
        if arg.ndim == 2:
            arg = arg.mean(axis=1)
        arg = frame(arg, frame_length=frame_length, hop_length=hop_length,
                    window='boxcar')
        if i == 0:
            activity = 10*np.log10(np.sum(arg**2, axis=1) + 1e-10)
            activity = (activity - activity.max()) > -abs(DRdB)
        args[i] = arg[activity, :]
    scores = []
    for i in range(len(args)//2):
        ref, hat = args[2*i], args[2*i+1]
        if i == 0:
            alpha = np.sum(ref*hat, axis=1)/np.sum(ref**2, axis=1)
            alpha = alpha.reshape(-1, 1)
            if sdr == 'default':
                num = ref
                den = ref-hat
            elif sdr == 'si':
                num = alpha*ref
                den = alpha*ref-hat
            elif sdr == 'sd':
                num = alpha*ref
                den = ref-hat
            else:
                raise ValueError("`sdr` must be 'default', 'si' or 'sd'")
        else:
            num = ref
            den = hat
            # TODO: extend the definitions of SI-SDR and SD-SDR to segNR
        score = np.mean(10*np.log10(
            (np.sum(num**2, axis=1) + 1e-10) /
            (np.sum(den**2, axis=1) + 1e-10)
        ))
        scores.append(score)
    return scores
