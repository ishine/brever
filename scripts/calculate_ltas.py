import argparse

import numpy as np
import matplotlib.pyplot as plt

from brever.io import get_ltas


def main():
    ltas = get_ltas(args.speaker, verbose=True)

    plt.figure()
    plt.semilogx(10*np.log10(ltas))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate long-term average '
                                                 'spectrum')
    parser.add_argument('speaker', help='speaker regular expression, e.g. '
                                        'timit_.*')
    args = parser.parse_args()
    main()
