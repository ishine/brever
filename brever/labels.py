import numpy as np

eps = np.finfo(float).eps


def irm(target, masker, beta=0.5, time_axis=-1, channel_axis=-3):
    # calculate energy
    target = np.mean(target**2, axis=(time_axis, channel_axis), keepdims=True)
    masker = np.mean(masker**2, axis=(time_axis, channel_axis), keepdims=True)
    irm = (1 + masker/(target+eps))**-0.5
    return irm.squeeze(axis=(time_axis, channel_axis))
