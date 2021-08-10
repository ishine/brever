import os
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from brever.io import get_ltas


def main():
    ltas = get_ltas('timit_.*', verbose=True)

    plt.figure()
    plt.semilogx(10*np.log10(ltas))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate timit ltas')
    args = parser.parse_args()
    main()
