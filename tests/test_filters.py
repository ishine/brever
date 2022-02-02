import numpy as np

from brever.filters import Filterbank


def test_filterbank():
    filterbank = Filterbank()
    x = np.random.randn(2, 512)
    x = filterbank.filt(x)
    filterbank.rfilt(x)
