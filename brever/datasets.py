import os
import numpy as np
import scipy.signal
import soundfile as sf
from resampy import resample

from .utils import zero_pad


def spatialize(x, brir):
    x_left = scipy.signal.oaconvolve(x, brir[:, 0], mode='same')
    x_right = scipy.signal.oaconvolve(x, brir[:, 1], mode='same')
    return np.vstack([x_left, x_right]).T


def diffuse_noise(brir_dirpath, n_samples):
    noise = np.zeros((n_samples, 2))
    for filename in os.listdir(brir_dirpath):
        brir, _ = sf.read(os.path.join(brir_dirpath, filename))
        noise += spatialize(np.random.randn(n_samples), brir)
    return noise


def make_mixture(target_path, target_angle, brir_dirpath, snr, padding=0):
    for filename in os.listdir(brir_dirpath):
        if '_%ideg_' % target_angle in filename:
            brir_filepath = os.path.join(brir_dirpath, filename)
            break
    brir, fs = sf.read(brir_filepath)
    target, fs_old = sf.read(target_path)
    target = resample(target, fs_old, fs)
    target = spatialize(target, brir)
    n_padding = round(padding*fs)
    noise = diffuse_noise(brir_dirpath, len(target)+n_padding*2)
    noise_energy = np.sum(noise[n_padding:len(target)-n_padding]**2)
    target_energy = np.sum(target**2)
    noise *= 10**(-snr/10)*(target_energy/noise_energy)**0.5
    target = zero_pad(target, n_padding, 'both')
    return noise + target
