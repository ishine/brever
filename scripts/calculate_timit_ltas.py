import os
import argparse
import sys

import numpy as np
import soundfile as sf
import scipy.signal

from brever.config import defaults


def main(plot):
    config = defaults()
    timit_dirpath = config.PRE.MIXTURES.PATH.TIMIT
    all_filepaths = []
    for root, dirs, files in os.walk(timit_dirpath):
        for file in files:
            if (file.endswith(('.wav', '.WAV')) and 'SA1' not in file and 'SA2'
                    not in file):
                all_filepaths.append(os.path.join(root, file))

    n = 257
    ltas = np.zeros(n)

    for i, filepath in enumerate(all_filepaths):
        sys.stdout.write('\r')
        sys.stdout.write(f'Processing recording {i+1}/{len(all_filepaths)}')
        x, _ = sf.read(filepath)
        _, _, X = scipy.signal.stft(x, nperseg=512, noverlap=256)
        ltas += np.mean(np.abs(X)**2, axis=1)
    sys.stdout.write('\n')

    n_oct = 3
    f = np.arange(1, n)
    sigma = (f/n_oct)/np.pi
    df = np.subtract.outer(f, f)
    g = np.exp(-0.5*(df/sigma)**2)/(sigma*(2*np.pi)**0.5)
    g /= g.sum(axis=1)
    ltas_smooth = np.copy(ltas)
    ltas_smooth[1:] = g@ltas_smooth[1:]
    np.save('ltas.npy', ltas_smooth)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(g)
        plt.figure()
        plt.plot(g)
        plt.figure()
        plt.semilogx(10*np.log10(ltas))
        plt.semilogx(10*np.log10(ltas_smooth))
        plt.grid()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate timit ltas')
    parser.add_argument('--plot', action='store_true',
                        help='plot ltas and smoothing function')
    args = parser.parse_args()
    main(args.plot)
