import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import to_rgb
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
            pass
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


def get_color_cycle():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def barplot(datas, ax, xticklabels=None, errs=None, ylabel=None, labels=None,
            colors=None, rotation=None, ha='right', lw=None):
    # check datas type
    if isinstance(datas, (float, int)):
        datas = np.array(datas)
    if isinstance(datas, np.ndarray):
        if datas.ndim == 0:
            datas = np.array(datas)
        if datas.ndim == 1:
            datas = datas[np.newaxis, :]
        if datas.ndim == 2:
            datas = [datas[:, i] for i in range(datas.shape[1])]
        elif datas.ndim == 3:
            datas = [datas[i, :, :] for i in range(datas.shape[0])]
        else:
            raise ValueError('barplot input as a numpy array must be at most '
                             f'3D, got {datas.ndim}D')
    elif not isinstance(datas, list):
        raise ValueError('barplot input must be a numpy array or a list of '
                         'numpy arrays')
    if len(datas) == 0:
        raise ValueError('cannot barplot empty data')
    for i, data in enumerate(datas):
        if not isinstance(data, np.ndarray):
            raise ValueError('barplot input must be a numpy array or a list '
                             'of numpy arrays')
        if data.size == 0:
            raise ValueError(f'barplot input at position {i} is empty')
        if data.ndim == 1:
            datas[i] = data[np.newaxis, :]
        if data.ndim > 2:
            raise ValueError(f'barplot input at position {i} is {data.ndim}D, '
                             'must be at most 2D')
        if datas[i].shape[0] != datas[0].shape[0]:
            raise ValueError('all barplot inputs must have the same size '
                             'along first dimension')
    # check errs type
    if errs is not None:
        if isinstance(errs, (float, int)):
            errs = np.array(errs)
        if isinstance(errs, np.ndarray):
            if errs.ndim == 0:
                errs = np.array(errs)
            if errs.ndim == 1:
                errs = errs[np.newaxis, :]
            if errs.ndim == 2:
                errs = [errs[:, i] for i in range(errs.shape[1])]
            elif errs.ndim == 3:
                errs = [errs[i, :, :] for i in range(errs.shape[0])]
            else:
                raise ValueError('errs as a numpy array must be at most '
                                 f'3D, got {errs.ndim}D')
        elif not isinstance(errs, list):
            raise ValueError('errs must be a numpy array or a list of '
                             'numpy arrays')
        if len(errs) != len(datas):
            raise ValueError(f'datas and errs must have the same length, got '
                             f'{len(datas)} and {len(errs)}')
        for i, (data, err) in enumerate(zip(datas, errs)):
            if err.ndim == 1:
                errs[i] = err[np.newaxis, :]
            if errs[i].shape != data.shape:
                raise ValueError('elements of datas and errs must have same '
                                 f'shape pair-wise, got shapes {data.shape} '
                                 f'and {errs[i].shape} at position {i}')
    # check labels
    if colors is not None and len(colors) != len(datas):
        raise ValueError('colors must have the same length as datas')
    # main
    color_cycle = get_color_cycle()
    n_conditions = datas[0].shape[0]
    n_models = sum(data.shape[1] for data in datas)
    if labels is not None and len(labels) != n_models:
        print(f'Warning: the number of labels ({len(labels)}) does not match '
              f'the total number of models ({n_models})')
    bar_width = 1/(n_models+1)
    model_count = 0
    patches = []
    for i, data in enumerate(datas):
        if colors is None:
            color = color_cycle[i % len(color_cycle)]
        else:
            color = colors[i]
        for j in range(data.shape[1]):
            color = to_rgb(color)
            color = color + (1-j/data.shape[1], )
            offset = (model_count - (n_models-1)/2)*bar_width
            x = np.arange(n_conditions) + offset
            if errs is None:
                yerr = None
            else:
                yerr = errs[i][:, j]
            if labels is None:
                label = None
            else:
                try:
                    label = labels[model_count]
                except IndexError:
                    label = ''
            patch = ax.bar(x, data[:, j], width=bar_width, color=color,
                           yerr=yerr, label=label,
                           error_kw={'lw': lw})
            model_count += 1
            patches.append(patch)
    if xticklabels is None:
        xticklabels = np.arange(n_conditions)
    ax.set_xticks(np.arange(n_conditions))
    if rotation is None:
        ha = 'center'
    ax.set_xticklabels(xticklabels, rotation=rotation, ha=ha)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='dotted')
    ax.set_axisbelow(True)
    return patches
