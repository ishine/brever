import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import Colorbar
import numpy as np


class TickFormatter:
    def __init__(self, x, y, ax=None):
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.x = x
        self.y = y
        self.showed = False
        ax.figure.canvas.mpl_connect('draw_event', self)
        ax.figure.canvas.mpl_connect('resize_event', self)

    def __call__(self, event):
        if self.ax.figure._cachedRenderer is None:
            return
        if event.name == 'draw_event' and self.showed:
            return
        self.showed = True
        self.ax.xaxis.set_major_locator(ticker.AutoLocator())
        self.ax.yaxis.set_major_locator(ticker.AutoLocator())
        xticks = self.ax.get_xticks().astype(int)
        yticks = self.ax.get_yticks().astype(int)
        xticks = xticks[(0 <= xticks) & (xticks < len(self.x))]
        yticks = yticks[(0 <= yticks) & (yticks < len(self.y))]
        xticklabels = self.x[xticks]
        yticklabels = self.y[yticks].round().astype(int)
        self.ax.set(
            xticks=xticks,
            yticks=yticks,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        self.ax.figure.tight_layout()


def plot_spectrogram(X, ax=None, fs=1, hop_length=1, f=None, imshow_kw={},
                     cbar_kw={}, set_kw={}):
    if ax is None:
        ax = plt.gca()
    if f is None:
        f = np.arange(X.shape[1])
    # create x-data
    t = np.arange(X.shape[0])*hop_length/fs
    if fs == 1:
        t = t.astype(int)
    # plot
    im = ax.imshow(X.T, aspect='auto', origin='lower', **imshow_kw)
    # set axis properties
    default_set_kw = {
        'xlabel': 'Time (s)',
        'ylabel': 'Frequency (Hz)',
    }
    set_kw = {**default_set_kw, **set_kw}
    ax.set(**set_kw)
    # initialize tick formatter
    TickFormatter(ax=ax, x=t, y=f)
    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    return im, cbar


def plot_waveform(x, fs=1, ax=None, plot_kw={}, set_kw={}):
    if ax is None:
        ax = plt.gca()
    # create x-data
    t = np.arange(len(x))/fs
    if fs == 1:
        t = t.astype(int)
    # plot
    default_plot_kw = {
        'linewidth': 1,
    }
    plot_kw = {**default_plot_kw, **plot_kw}
    line = ax.plot(t, x, **plot_kw)
    # set axis properties
    default_set_kw = {
        'xlabel': 'Time (s)',
        'ylabel': 'Amplitude',
        'xlim': (0, len(x)/fs),
    }
    set_kw = {**default_set_kw, **set_kw}
    ax.set(**set_kw)
    return line


def get_colorbars(fig, _cbars=[]):
    for child in fig.get_children():
        if isinstance(getattr(child, 'colorbar', None), Colorbar):
            _cbars.append(child.colorbar)
        get_colorbars(child, _cbars)
    return _cbars


def share_xlim(axes):
    xmin, xmax = axes[0].get_xlim()
    for ax in axes:
        xmin_, xmax_ = ax.get_xlim()
        xmin = min(xmin, xmin_)
        xmax = max(xmax, xmax_)
    for ax in axes:
        ax.set_xlim(xmin, xmax)


def share_ylim(axes, center_zero=False):
    ymin, ymax = axes[0].get_ylim()
    if center_zero:
        ymax = max(abs(ymin), abs(ymax))
        ymin = -ymax
    for ax in axes:
        ymin_, ymax_ = ax.get_ylim()
        if center_zero:
            ymax_ = max(abs(ymin_), abs(ymax_))
            ymin_ = -ymax_
        ymin = min(ymin, ymin_)
        ymax = max(ymax, ymax_)
    for ax in axes:
        ax.set_ylim(ymin, ymax)


def share_clim(axes):
    cmin, cmax = axes[0].images[0].get_clim()
    for ax in axes:
        cmin_, cmax_ = ax.images[0].get_clim()
        cmin = min(cmin, cmin_)
        cmax = max(cmax, cmax_)
    for ax in axes:
        ax.images[0].set_clim(cmin, cmax)
