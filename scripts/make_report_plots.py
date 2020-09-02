import os
import itertools
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import yaml
import torch
import soundfile as sf
import matlab
import matlab.engine

from brever.features import itd, ic, ild, pdf, mfcc
from brever.modelmanagement import get_file_indices, get_feature_indices
from brever.filters import mel_filt, mel_triangle_filterbank
from brever.utils import frame, wola
from brever.config import defaults
from brever.pytorchtools import Feedforward, H5Dataset, TensorStandardizer

from compare_models import main
from compare_regularization import main as main_reg
from compare_target_location import main as main_loc


FONT_SIZE = 7
TITLE_SIZE = 7
plt.rcParams['axes.titlesize'] = TITLE_SIZE
plt.rcParams['axes.labelsize'] = FONT_SIZE
plt.rcParams['xtick.labelsize'] = FONT_SIZE
plt.rcParams['ytick.labelsize'] = FONT_SIZE
plt.rcParams['legend.fontsize'] = FONT_SIZE
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['patch.facecolor'] = 'b'
plt.rcParams['patch.linewidth'] = .5
plt.rcParams['axes.linewidth'] = .5
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['hatch.linewidth'] = .5
plt.rcParams['xtick.major.width'] = .5
plt.rcParams['xtick.minor.width'] = .5
plt.rcParams['ytick.major.width'] = .5
plt.rcParams['ytick.minor.width'] = .5
plt.rcParams['lines.linewidth'] = .5
plt.rcParams['lines.markeredgewidth'] = .5

plt.ion()

dirpath = r'C:\Users\Philippe\Desktop\brever-temp\svgs'
scaling = .5


def heatmap(data, row_labels, col_labels, ax=None, fig=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels, rotation='90', va='center')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="both", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def remove_patches(fig, axes):
    for ax in axes:
        ax.patch.set_visible(False)
    fig.patch.set_visible(False)


def set_yaxis(axes, tag, yticks=None):
    for ax in axes:
        if tag == 'pesq':
            ax.set_ylabel(r'\$\Delta\$PESQ')
            if yticks is not None:
                ax.set_yticks(yticks)


def set_xaxis(axes):
    axes[1].set_xticklabels(['A', 'B', 'C', 'D', 'mean'])
    axes[1].set_xlabel('Room')


def make_legend(axes, handles, labels, ncol):
    handles = flip(handles, ncol)
    labels = flip(labels, ncol)
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    axes[1].legend(handles, labels, loc='lower center', ncol=ncol,
                   bbox_to_anchor=(-.1, 1))


def save_figure(fig, basename, tag):
    filename = f'{basename}_{tag}.svg'
    filepath = os.path.join(dirpath, filename)
    fig.savefig(filepath)


def experiment_regularization():
    main_reg(
        layers=None,
        stacks=4,
        features=None,
        batchsize=None,
        centered=False,
        onlyreverb=False,
        testbig=True,
        trainbig=False,
    )
    curfignum = plt.gcf().number
    for i, tag in zip([curfignum-1, curfignum], ['mse', 'pesq']):
        fig = plt.figure(i)
        axes = fig.axes
        set_yaxis(axes, tag)
        set_xaxis(axes)
        labels = ['None', 'Dropout', 'Batchnorm', 'Both']
        handles, _ = axes[1].get_legend_handles_labels()
        make_legend(axes, handles, labels, ncol=4)
        fig.set_size_inches(15*scaling, 4*scaling)
        remove_patches(fig, axes)
        if tag == 'pesq':
            for ax in axes:
                ax.set_yticks(np.arange(0, 0.8, 0.1))
        save_figure(fig, 'exp_regularization', tag)


def experiment_context():
    main(
        dimension='stacks',
        layers=None,
        stacks=4,
        batchnorm=None,
        dropout=True,
        features=None,
        batchsize=None,
        centered=False,
        onlyreverb=False,
        testbig=True,
        trainbig=False,
    )
    curfignum = plt.gcf().number
    for i, tag in zip([curfignum-1, curfignum], ['mse', 'pesq']):
        fig = plt.figure(i)
        axes = fig.axes
        set_yaxis(axes, tag)
        set_xaxis(axes)
        labels = [
            u'1 frame (32 ms \u2014 no stacking)',
            '2 frames (48 ms)',
            '3 frames (64 ms)',
            '4 frames (80 ms)',
            '5 frames (96 ms)',
        ]
        handles, _ = axes[1].get_legend_handles_labels()
        make_legend(axes, handles, labels, ncol=3)
        fig.set_size_inches(15*scaling, 4*scaling)
        remove_patches(fig, axes)
        if tag == 'pesq':
            for ax in axes:
                ax.set_yticks(np.arange(0, 0.8, 0.1))
        save_figure(fig, 'exp_stacks', tag)


def experiment_layers():
    main(
        dimension='layers',
        layers=None,
        stacks=4,
        batchnorm=None,
        dropout=True,
        features=None,
        batchsize=None,
        centered=False,
        onlyreverb=False,
        testbig=True,
        trainbig=False,
    )
    curfignum = plt.gcf().number
    for i, tag in zip([curfignum-1, curfignum], ['mse', 'pesq']):
        fig = plt.figure(i)
        axes = fig.axes
        set_yaxis(axes, tag)
        set_xaxis(axes)
        labels = [
            '1 hidden layer',
            '2 hidden layers',
            '3 hidden layers',
        ]
        handles, _ = axes[1].get_legend_handles_labels()
        make_legend(axes, handles, labels, ncol=3)
        fig.set_size_inches(15*scaling, 4*scaling)
        remove_patches(fig, axes)
        if tag == 'pesq':
            for ax in axes:
                ax.set_yticks(np.arange(0, 0.8, 0.1))
        save_figure(fig, 'exp_layers', tag)


def experiment_features():
    main(
        dimension='features',
        layers=None,
        stacks=4,
        batchnorm=None,
        dropout=True,
        features=None,
        batchsize=None,
        centered=False,
        onlyreverb=False,
        testbig=True,
        trainbig=False,
    )
    curfignum = plt.gcf().number
    for i, tag in zip([curfignum-1, curfignum], ['mse', 'pesq']):
        fig = plt.figure(i)
        axes = fig.axes
        set_yaxis(axes, tag)
        set_xaxis(axes)
        handles, labels = axes[1].get_legend_handles_labels()
        labels = [label[12:-1] for label in labels]
        labels = [label.replace("'", '') for label in labels]
        labels = [label.upper() for label in labels]
        make_legend(axes, handles, labels, ncol=4)
        fig.set_size_inches(15*scaling, 4*scaling)
        remove_patches(fig, axes)
        if tag == 'pesq':
            for ax in axes:
                ax.set_yticks(np.arange(0, 0.8, 0.1))
        save_figure(fig, 'exp_features', tag)


def experiment_target_location(onlyreverb=False, onlydiffuse=False,
                               testbig=True):
    if onlyreverb and onlydiffuse:
        raise ValueError('onlyreverb and onlydiffuse can\'t both be True')
    for testcenter, tag_ in zip([False, True], ['random', 'fixed']):
        main_loc(
            layers=1,
            stacks=4,
            batchnorm=None,
            dropout=True,
            features=None,
            batchsize=None,
            testcenter=testcenter,
            testbig=testbig,
            onlyreverb=onlyreverb,
            onlydiffuse=onlydiffuse,
        )
        curfignum = plt.gcf().number
        for i, tag in zip([curfignum-1, curfignum], ['mse', 'pesq']):
            fig = plt.figure(i)
            axes = fig.axes
            set_yaxis(axes, tag)
            set_xaxis(axes)
            handles, labels = axes[1].get_legend_handles_labels()
            labels = [label.replace("'", '') for label in labels]
            labels = [label.replace("[", '') for label in labels]
            labels = [label.replace("]", '') for label in labels]
            labels = [label.replace("centered_training", 'Fixed location')
                      for label in labels]
            labels = [label.replace("training", 'Random location')
                      for label in labels]
            labels = [label.replace("_", u' \u2014 ') for label in labels]
            labels = [label.replace("ic", 'IC') for label in labels]
            labels = [label.replace("ild", 'ILD') for label in labels]
            labels = [label.replace("itd", 'ITD') for label in labels]
            labels = [label.replace("mfcc", 'MFCC') for label in labels]
            labels = [label.replace("pdf", 'PDF') for label in labels]
            make_legend(axes, handles, labels, ncol=2)
            fig.set_size_inches(15*scaling, 4*scaling)
            remove_patches(fig, axes)
            if onlyreverb:
                fulltag = f'onlyreverb_{tag_}_{tag}'
            elif onlydiffuse:
                fulltag = f'onlydiffuse_{tag_}_{tag}'
            else:
                fulltag = f'{tag_}_{tag}'
            if testbig and not onlyreverb and not onlydiffuse:
                if testcenter:
                    if tag == 'pesq':
                        for ax in axes:
                            ax.set_yticks(np.arange(0, 0.9, 0.1))
                else:
                    if tag == 'pesq':
                        for ax in axes:
                            ax.set_yticks(np.arange(-0.1, 0.8, 0.1))
            save_figure(fig, 'exp_location', fulltag)


def experiment_dataset_size():
    models_sorted = [
        ['7eee8c945c86168f0dfa8a8c5a122b6b23c3d7cb1c940c752a4abe7ce06231bd'],
        ['a9f53cd876d48a7a72edeecbb3a58a31e2fa50f75239bf8cf90a43afa4f8f3e5'],
    ]
    pesqs = []
    mses = []
    for models in models_sorted:
        pesq = []
        mse = []
        for model_id in models:
            pesq_filepath = os.path.join('models', model_id, 'eval_PESQ.npy')
            mse_filepath = os.path.join('models', model_id, 'eval_MSE.npy')
            pesq.append(np.load(pesq_filepath))
            mse.append(np.load(mse_filepath))
            with open(os.path.join('models', model_id, 'config.yaml')) as f:
                print(yaml.safe_load(f))
        pesqs.append(np.asarray(pesq).mean(axis=0))
        mses.append(np.asarray(mse).mean(axis=0))
    snrs = [0, 3, 6, 9, 12, 15]
    room_names = ['A', 'B', 'C', 'D']
    labels = [
        '1000 mixtures',
        '10000 mixtures',
    ]
    n = len(pesqs)
    width = 1/(n+1)
    print(f'Comparing {len(models_sorted)} group(s) of models')
    for i, models in enumerate(models_sorted):
        print((f'Group {i+1} contains {len(models_sorted[i])} model(s):'))
        for model in models:
            print(f'  {model}')
    for metrics, ylabel, filetag in zip(
                [mses, pesqs],
                ['MSE', r'\$\Delta\$PESQ'],
                ['mse', 'pesq'],
            ):
        fig, axes = plt.subplots(1, 2, sharey=True)
        for axis, (ax, xticklabels, xlabel, filetag_) in enumerate(zip(
                    axes[::-1],
                    [room_names, snrs],
                    ['Room', 'SNR (dB)'],
                    ['rooms', 'snrs'],
                )):
            for i, (metric, label) in enumerate(zip(metrics, labels)):
                data = metric.mean(axis=axis)
                data = np.hstack((data, data.mean()))
                x = np.arange(len(data)) + (i - (n-1)/2)*width
                x[-1] = x[-1] + 2*width
                ax.bar(
                    x=x,
                    height=data,
                    width=width,
                    label=label,
                )
            xticks = np.arange(len(xticklabels) + 1, dtype=float)
            xticks[-1] = xticks[-1] + 2*width
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels + ['mean'])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.yaxis.set_tick_params(labelleft=True)
            if filetag == 'pesq':
                ax.set_yticks(np.arange(0.0, 0.8, 0.1))
        fig.tight_layout()
        axes[0].get_legend().remove()
        axes[1].legend(labels, loc='lower center',
                       bbox_to_anchor=(-.1, 1), ncol=2)
        fig.set_size_inches(15*scaling, 4*scaling)
        remove_patches(fig, axes)
        save_figure(fig, 'exp_datasetsize', filetag)
    plt.show()


def experiment_baseline():
    model_id = 'a9f53cd876d48a7a72edeecbb3a58a31e2fa50f75239bf8cf90a43afa4f8f3e5'
    with open(os.path.join('models', model_id, 'config.yaml')) as f:
        print(yaml.safe_load(f))
    pesqs = []

    baseline_scores_dir = r'matlab\scores'
    snrs = [0, 3, 6, 9, 12, 15]
    room_names = ['A', 'B', 'C', 'D']
    baseline_pesqs = np.zeros((len(snrs), len(room_names)))
    for i, snr in enumerate(snrs):
        for j, room in enumerate(room_names):
            filename = f'testing_big_snr{snr}_room{room}.npy'
            filepath = os.path.join(baseline_scores_dir, filename)
            score = np.load(filepath)
            baseline_pesqs[i, j] = score
    pesqs.append(baseline_pesqs)

    pesq_filepath = os.path.join('models', model_id, 'eval_PESQ.npy')
    pesqs.append(np.load(pesq_filepath))

    labels = [
        u'D\u00F6rbecker and Ernst',
        'Neural network --- MFCC, IC',
    ]
    n = len(pesqs)
    width = 1/(n+1)
    for metrics, ylabel, filetag in zip(
                [pesqs],
                [r'\$\Delta\$PESQ'],
                ['pesq'],
            ):
        fig, axes = plt.subplots(1, 2, sharey=True)
        for axis, (ax, xticklabels, xlabel, filetag_) in enumerate(zip(
                    axes[::-1],
                    [room_names, snrs],
                    ['Room', 'SNR (dB)'],
                    ['rooms', 'snrs'],
                )):
            for i, (metric, label) in enumerate(zip(metrics, labels)):
                data = metric.mean(axis=axis)
                data = np.hstack((data, data.mean()))
                x = np.arange(len(data)) + (i - (n-1)/2)*width
                x[-1] = x[-1] + 2*width
                ax.bar(
                    x=x,
                    height=data,
                    width=width,
                    label=label,
                )
                print(data.round(2))
            xticks = np.arange(len(xticklabels) + 1, dtype=float)
            xticks[-1] = xticks[-1] + 2*width
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels + ['mean'])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.yaxis.set_tick_params(labelleft=True)
            if filetag == 'pesq':
                ax.set_yticks(np.arange(0.0, 0.8, 0.1))
        fig.tight_layout()
        axes[0].get_legend().remove()
        axes[1].legend(labels, loc='lower center',
                       bbox_to_anchor=(-.1, 1), ncol=2)
        fig.set_size_inches(15*scaling, 4*scaling)
        remove_patches(fig, axes)
        save_figure(fig, 'exp_baseline', filetag)
    plt.show()


def experiment_final_heatmap():
    model = 'a9f53cd876d48a7a72edeecbb3a58a31e2fa50f75239bf8cf90a43afa4f8f3e5'
    pesq_filepath = os.path.join('models', model, 'eval_PESQ.npy')
    pesqs = np.load(pesq_filepath)
    pesqs = np.hstack((pesqs, pesqs.mean(axis=1, keepdims=True)))
    pesqs = np.vstack((pesqs, pesqs.mean(axis=0, keepdims=True)))
    rooms = ['A', 'B', 'C', 'D', 'mean']
    snrs = [0, 3, 6, 9, 12, 15, 'mean']
    fig, ax = plt.subplots()
    im, cbar = heatmap(pesqs.T, rooms, snrs, ax=ax, fig=fig, cmap='Blues',
                       cbarlabel=r'\$\Delta\$PESQ', origin='lower')
    annotate_heatmap(im, valfmt="{x:.2f}")
    ax.grid(False)
    ax.set_xlabel('SNR(dB)')
    ax.set_ylabel('Room')
    ax.tick_params(axis='both', which='major', pad=0)
    cbar.ax.tick_params(axis='both', which='major', pad=2, length=3)
    fig.tight_layout()
    g = 0.70
    fig.set_size_inches(4.48*g, 3.36*g)
    remove_patches(fig, [ax])
    filepath = os.path.join(dirpath, f'exp_final_heatmap.svg')
    fig.savefig(filepath)
    plt.show()


def experiment_delta_delta_pesq():
    model_spectral = 'a409fa2e3860a33f33234b52c9a4de6e651875fe5c49f510f0890507dec089ef'
    model_combined = '7eee8c945c86168f0dfa8a8c5a122b6b23c3d7cb1c940c752a4abe7ce06231bd'

    pesq_filepath = os.path.join('models', model_spectral, 'eval_PESQ.npy')
    pesqs = np.load(pesq_filepath)
    pesqs = np.hstack((pesqs, pesqs.mean(axis=1, keepdims=True)))
    pesqs_spectral = np.vstack((pesqs, pesqs.mean(axis=0, keepdims=True)))

    pesq_filepath = os.path.join('models', model_combined, 'eval_PESQ.npy')
    pesqs = np.load(pesq_filepath)
    pesqs = np.hstack((pesqs, pesqs.mean(axis=1, keepdims=True)))
    pesqs_combined = np.vstack((pesqs, pesqs.mean(axis=0, keepdims=True)))

    pesqs = pesqs_combined - pesqs_spectral

    rooms = ['A', 'B', 'C', 'D', 'mean']
    snrs = [0, 3, 6, 9, 12, 15, 'mean']
    fig, ax = plt.subplots()
    im, cbar = heatmap(pesqs.T, rooms, snrs, ax=ax, fig=fig, cmap='Blues',
                       cbarlabel=r'\$\Delta\$PESQ', origin='lower')
    annotate_heatmap(im, valfmt="{x:.2f}")
    ax.grid(False)
    ax.set_xlabel('SNR(dB)')
    ax.set_ylabel('Room')
    ax.tick_params(axis='both', which='major', pad=0)
    cbar.ax.tick_params(axis='both', which='major', pad=2, length=3)
    fig.tight_layout()
    g = 0.70
    fig.set_size_inches(4.48*g, 3.36*g)
    remove_patches(fig, [ax])
    filepath = os.path.join(dirpath, f'exp_delta_delta.svg')
    fig.savefig(filepath)
    plt.show()


def example_specgrams():
    ds_dir = r'data\processed\centered_testing_snr15_roomD'
    ds_filepath = os.path.join(ds_dir, 'dataset.hdf5')

    mix_index = 8
    file_indices = get_file_indices(ds_dir)
    i_start, i_end = file_indices[mix_index]
    with h5py.File(ds_filepath, 'r') as f:
        IRM = f['labels'][i_start:i_end]
        mix = f['mixtures'][mix_index].reshape(-1, 2)
        foreground = f['foregrounds'][mix_index].reshape(-1, 2)
        background = f['backgrounds'][mix_index].reshape(-1, 2)

    mix_filt, fc = mel_filt(mix)
    mix_filt_framed = frame(mix_filt)
    MIX = 10*np.log10((mix_filt_framed**2).mean(axis=(1, 3)))

    foreground_filt, fc = mel_filt(foreground)
    foreground_filt_framed = frame(foreground_filt)
    FORE = 10*np.log10((foreground_filt_framed**2 + 1e-10).mean(axis=(1, 3)))

    background_filt, fc = mel_filt(background)
    background_filt_framed = frame(background_filt)
    BACK = 10*np.log10((background_filt_framed**2).mean(axis=(1, 3)))

    ILD = ild(mix_filt_framed, filtered=True, framed=True)
    ITD = itd(mix_filt_framed, filtered=True, framed=True)
    IC = ic(mix_filt_framed, filtered=True, framed=True)
    PDF = pdf(mix_filt_framed, filtered=True, framed=True)
    MFCC = mfcc(mix_filt_framed, filtered=True, framed=True)

    PDF = np.log(PDF)
    MFCC = (MFCC - MFCC.mean(axis=0))/MFCC.std(axis=0)

    scaling_ = 0.525
    fig, axes = plt.subplots(3, 3, figsize=(15*scaling_, 9.875*scaling_))

    datas = [[MIX, FORE, BACK], [IRM, ILD, ITD], [IC, PDF, MFCC]]
    titles = [
        ['Mixture', 'Foreground', 'Background'],
        ['IRM', 'ILD', 'ITD'],
        ['IC', 'log-PDF', 'MFCC'],
    ]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            data = datas[i][j]
            title = titles[i][j]

            pos = ax.imshow(data.T, origin='lower')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.5%", pad=0.05)
            cbar = fig.colorbar(pos, cax=cax)

            xticklabels = np.arange(0, 1.5, 0.25)
            xticks = xticklabels*16e3/256
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks*256/16e3)
            ax.set_xlabel('Time (s)')

            if title != 'MFCC':
                yticks = np.arange(0, 70, 15)
                ax.set_yticks(yticks)
                ax.set_yticklabels(fc[yticks].round().astype(int))
                ax.set_ylabel('Frequency (Hz)')
            else:
                ax.set_ylabel('Order')
                cmin, cmax = np.quantile(data, [0.01, 0.99])
                ax.get_images()[0].set_clim(cmin, cmax)
            # if title == 'ILD':
            #     cmin, cmax = np.quantile(data, [0.005, 0.995])
            #     ax.get_images()[0].set_clim(cmin, cmax)
            if title == 'IRM':
                ax.get_images()[0].set_clim(0, 1)
            if i == 0:
                cmin = min(MIX.min(), FORE.min(), BACK.min())
                cmax = min(MIX.max(), FORE.max(), BACK.max())
                ax.get_images()[0].set_clim(cmin, cmax)

            ax.set_title(title)
            ax.grid(False)

            pos = ax.title.get_position()
            ax.title.set_position([pos[0], pos[1]-0.04])
            ax.xaxis.labelpad = 2
            ax.yaxis.labelpad = 2
            ax.tick_params(axis='both', which='major', pad=2, length=3)
            cbar.ax.tick_params(axis='both', which='major', pad=2, length=3)

    remove_patches(fig, axes.ravel())
    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.45,
        hspace=0.4,
    )

    filepath = os.path.join(dirpath, 'features.svg')
    fig.savefig(filepath)
    plt.show()


def example_enhancement(mix_index, test_dataset_dir=None, save_filepath=None,
                        matlab_engine=None, layout='report', foreback=False):
    if foreback and layout != 'ppt':
        raise ValueError('Cannot toggle foreback if layout is not ppt')

    model_id = 'a9f53cd876d48a7a72edeecbb3a58a31e2fa50f75239bf8cf90a43afa4f8f3e5'
    model_dir = os.path.join('models', model_id)

    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    stat_path = os.path.join(model_dir, 'statistics.npy')
    mean, std = np.load(stat_path)

    if test_dataset_dir is None:
        test_dataset_dir = r'data\processed\centered_testing_snr6_roomA'
    test_dataset_path = os.path.join(test_dataset_dir, 'dataset.hdf5')
    feature_indices = get_feature_indices(test_dataset_dir,
                                          sorted(config.POST.FEATURES))
    file_indices = get_file_indices(test_dataset_dir)
    test_dataset = H5Dataset(
        filepath=test_dataset_path,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        # decimation=config.POST.DECIMATION,
        feature_indices=feature_indices,
        file_indices=file_indices,
    )

    if config.POST.GLOBALSTANDARDIZATION:
        test_dataset.transform = TensorStandardizer(mean, std)

    model = Feedforward(
        input_size=test_dataset.n_features,
        output_size=test_dataset.n_labels,
        n_layers=config.MODEL.NLAYERS,
        dropout_toggle=config.MODEL.DROPOUT.ON,
        dropout_rate=config.MODEL.DROPOUT.RATE,
        batchnorm_toggle=config.MODEL.BATCHNORM.ON,
        batchnorm_momentum=config.MODEL.BATCHNORM.MOMENTUM,
    )
    if config.MODEL.CUDA:
        model = model.cuda()
    state_file = os.path.join(model_dir, 'checkpoint.pt')
    model.load_state_dict(torch.load(state_file))

    pipes_file = os.path.join(config.POST.PATH.TRAIN, 'pipes.pkl')
    with open(pipes_file, 'rb') as f:
        pipes = pickle.load(f)
    scaler = pipes['scaler']
    filterbank = pipes['filterbank']
    framer = pipes['framer']

    i_start, i_end = file_indices[mix_index]

    features, IRM = test_dataset[i_start:i_end]
    if config.MODEL.CUDA:
        features = features.cuda()
    model.eval()
    with torch.no_grad():
        PRM = model(features)
        if config.MODEL.CUDA:
            PRM = PRM.cpu()
        PRM = PRM.numpy()

    with h5py.File(test_dataset_path, 'r') as f:
        mixture = f['mixtures'][mix_index].reshape(-1, 2)
        foreground = f['foregrounds'][mix_index].reshape(-1, 2)
        background = f['backgrounds'][mix_index].reshape(-1, 2)

    # scale signal
    scaler.fit(mixture)
    mixture = scaler.scale(mixture)
    foreground = scaler.scale(foreground)
    background = scaler.scale(background)
    scaler.__init__(scaler.active)

    # apply filterbank
    mixture_filt = filterbank.filt(mixture)
    foreground_filt = filterbank.filt(foreground)
    background_filt = filterbank.filt(background)

    # frame
    mixture_filt_framed = framer.frame(mixture_filt)
    foreground_filt_framed = framer.frame(foreground_filt)
    background_filt_framed = framer.frame(background_filt)

    MIX_I = (mixture_filt_framed**2).mean(axis=(1, 3))
    MIX_O = MIX_I*PRM
    MIX_I = 10*np.log10(MIX_I)
    MIX_O = 10*np.log10(MIX_O)

    # apply predicted RM
    PRM_ = wola(PRM, trim=len(mixture_filt))[:, :, np.newaxis]
    IRM_ = wola(IRM, trim=len(mixture_filt))[:, :, np.newaxis]
    mixture_enhanced = filterbank.rfilt(mixture_filt*PRM_)
    mixture_best = filterbank.rfilt(mixture_filt*IRM_)
    mixture_ref = filterbank.rfilt(mixture_filt)
    foreground_ref = filterbank.rfilt(foreground_filt)
    background_ref = filterbank.rfilt(background_filt)

    FORE = (foreground_filt_framed**2).mean(axis=(1, 3))
    BACK = (background_filt_framed**2).mean(axis=(1, 3))
    FORE = 10*np.log10(FORE + 1e-10)
    BACK = 10*np.log10(BACK + 1e-10)

    # start MATLAB engine
    if matlab_engine is None:
        print('Starting MATLAB engine...')
        matlab_engine = matlab.engine.start_matlab()
        paths = [
            r'matlab',
            r'matlab\loizou',
            r'matlab\stft-framework\stft-framework\src\tools',
        ]
        for path in paths:
            matlab_engine.addpath(path, nargout=0)

    # convert to matlab float
    mixture_ref = matlab.single(mixture_ref.tolist())

    # call baseline model
    P = matlab_engine.configSTFT(
        float(config.PRE.FS),
        32e-3,
        0.5,
        'hann',
        'wola',
    )
    mixture_baseline, G = matlab_engine.noiseReductionDoerbecker(
        mixture_ref,
        float(config.PRE.FS),
        P,
        nargout=2,
    )

    # convert back to numpy array
    mixture_ref = np.array(mixture_ref)
    mixture_baseline = np.array(mixture_baseline)

    # average gain and group FFT bins
    G = np.array(G).mean(axis=-1).T
    FB, _ = mel_triangle_filterbank()
    FB = FB/FB.sum(axis=0)
    G = G@FB

    # write audio
    if save_filepath is None:
        if layout == 'report':
            filename = f'enhancement_example.svg'
        elif layout == 'ppt':
            if foreback:
                filename = f'enhancement_example_foreback.svg'
            else:
                filename = f'enhancement_example_{layout}.svg'
        else:
            raise ValueError('Wrong layout!')
        save_filepath = os.path.join(dirpath, filename)
    audio_basename, _ = os.path.splitext(save_filepath)
    input_savepath = f'{audio_basename}_input.wav'
    output_savepath = f'{audio_basename}_output.wav'
    best_savepath = f'{audio_basename}_best.wav'
    baseline_savepath = f'{audio_basename}_baseline.wav'
    foreground_savepath = f'{audio_basename}_foreground.wav'
    background_savepath = f'{audio_basename}_background.wav'
    gain = 1/mixture.max()
    sf.write(input_savepath, mixture*gain, config.PRE.FS)
    sf.write(output_savepath, mixture_enhanced*gain, config.PRE.FS)
    sf.write(best_savepath, mixture_best*gain, config.PRE.FS)
    sf.write(baseline_savepath, mixture_baseline*gain, config.PRE.FS)
    sf.write(foreground_savepath, foreground_ref*gain, config.PRE.FS)
    sf.write(background_savepath, background_ref*gain, config.PRE.FS)

    # remove noise-only parts
    npad = round(config.PRE.MIXTURES.PADDING*config.PRE.FS)
    mixture_enhanced = mixture_enhanced[npad:-npad]
    mixture_ref = mixture_ref[npad:-npad]
    foreground_ref = foreground_ref[npad:-npad]
    mixture_baseline = mixture_baseline[npad:-npad]
    mixture_best = mixture_best[npad:-npad]

    # flatten and convert to matlab float
    mixture_enhanced = matlab.single(
        mixture_enhanced.sum(axis=1, keepdims=True).tolist())
    mixture_best = matlab.single(
        mixture_best.sum(axis=1, keepdims=True).tolist())
    mixture_ref = matlab.single(
        mixture_ref.sum(axis=1, keepdims=True).tolist())
    foreground_ref = matlab.single(
        foreground_ref.sum(axis=1, keepdims=True).tolist())
    mixture_baseline = matlab.single(
        mixture_baseline.sum(axis=1, keepdims=True).tolist())

    # calculate PESQ
    pesq_before = matlab_engine.pesq(foreground_ref, mixture_ref,
                                     config.PRE.FS)
    pesq_after = matlab_engine.pesq(foreground_ref, mixture_enhanced,
                                    config.PRE.FS)
    pesq_best = matlab_engine.pesq(foreground_ref, mixture_best,
                                   config.PRE.FS)
    pesq_baseline = matlab_engine.pesq(foreground_ref, mixture_baseline,
                                       config.PRE.FS)
    dpesq = pesq_after - pesq_before
    dpesq_best = pesq_best - pesq_before
    dpesq_baseline = pesq_baseline - pesq_before

    scaling_ = 0.525
    if layout == 'report':
        fig, axes = plt.subplots(5, 1, figsize=(15*scaling_, 16*scaling_))
        n_axes = 5
    elif layout == 'ppt':
        fig, axes = plt.subplots(2, 2, figsize=(17*scaling_, 5.75*scaling_))
        n_axes = 4
    else:
        raise ValueError('Wrong layout!')

    if foreback and layout == 'ppt':
        datas = [MIX_I, IRM, FORE, BACK]
        titles = [
            f'Input mixture (PESQ: {pesq_before:.2f})',
            rf'IRM (\$\Delta\$PESQ: +{dpesq_best:.2f})',
            'Foreground',
            'Background'
        ]
    else:
        datas = [MIX_I, IRM, PRM, G, MIX_O]
        doebeckernernst = u'D\u00F6rbecker and Ernst gain function'
        titles = [
            f'Input mixture (PESQ: {pesq_before:.2f})',
            rf'IRM (\$\Delta\$PESQ: +{dpesq_best:.2f})',
            rf'Neural network prediction (\$\Delta\$PESQ: +{dpesq:.2f})',
            rf'{doebeckernernst} (\$\Delta\$PESQ: +{dpesq_baseline:.2f})',
            f'Output mixture using the neural network prediction (PESQ: {pesq_after:.2f})',
        ]

    for i in range(n_axes):
        ax = axes.T.ravel()[i]
        data = datas[i]
        title = titles[i]

        pos = ax.imshow(data.T, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)
        cbar = fig.colorbar(pos, cax=cax)
        if i == 0:
            cmin, cmax = cbar.ax.get_ylim()
        elif i == 4 or foreback and i != 1:
            ax.get_images()[0].set_clim(cmin, cmax)

        _, xmax = ax.get_xlim()
        xticklabels = np.arange(0, xmax/16e3*256, 0.5)
        xticks = xticklabels*16e3/256
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Time (s)')

        yticks = np.arange(0, 70, 15)
        ax.set_yticks(yticks)
        fc = filterbank.fc
        ax.set_yticklabels(fc[yticks].round().astype(int))
        ax.set_ylabel('Frequency (Hz)')

        if i in [1, 2, 3] and not foreback:
            ax.get_images()[0].set_clim(0, 1)

        ax.set_title(title)
        ax.grid(False)

        pos = ax.title.get_position()
        ax.title.set_position([pos[0], pos[1]-0.04])
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis='both', which='major', pad=2, length=3)
        cbar.ax.tick_params(axis='both', which='major', pad=2, length=3)

    remove_patches(fig, axes.ravel())
    fig.subplots_adjust(
        # left=0.1,
        # right=0.9,
        # bottom=0.1,
        # top=0.9,
        # wspace=0.45,
        hspace=0.55,
    )

    fig.savefig(save_filepath)
    plt.show()

    print(f'MSE {((IRM-PRM)**2).mean()}')


def example_enhancement_each_condition(matlab_engine=None):
    if matlab_engine is None:
        print('Starting MATLAB engine...')
        matlab_engine = matlab.engine.start_matlab()
        matlab_engine.addpath('matlab\\loizou', nargout=0)
    for room in ['A', 'B', 'C', 'D']:
        for snr in [0, 3, 6, 9, 12, 15]:
            test_dataset_dir = f'data\\processed\\testing_snr{snr}_room{room}'
            save_dir = os.path.join(dirpath, 'enhancement_examples')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_filepath = os.path.join(save_dir, f'snr{snr}_room{room}.png')
            example_enhancement(0, test_dataset_dir=test_dataset_dir,
                                save_filepath=save_filepath,
                                matlab_engine=matlab_engine)
            plt.close()


def mse_vs_pesq_test(mix_index, test_dataset_dir=None, save_filepath=None,
                     matlab_engine=None, mode='vertical', layout='report'):
    model_id = 'a9f53cd876d48a7a72edeecbb3a58a31e2fa50f75239bf8cf90a43afa4f8f3e5'
    model_dir = os.path.join('models', model_id)

    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    if test_dataset_dir is None:
        test_dataset_dir = r'data\processed\centered_testing_snr6_roomA'
    test_dataset_path = os.path.join(test_dataset_dir, 'dataset.hdf5')
    feature_indices = get_feature_indices(test_dataset_dir,
                                          sorted(config.POST.FEATURES))
    file_indices = get_file_indices(test_dataset_dir)
    test_dataset = H5Dataset(
        filepath=test_dataset_path,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
        # decimation=config.POST.DECIMATION,
        feature_indices=feature_indices,
        file_indices=file_indices,
    )

    pipes_file = os.path.join(config.POST.PATH.TRAIN, 'pipes.pkl')
    with open(pipes_file, 'rb') as f:
        pipes = pickle.load(f)
    scaler = pipes['scaler']
    filterbank = pipes['filterbank']
    framer = pipes['framer']

    i_start, i_end = file_indices[mix_index]
    features, IRM = test_dataset[i_start:i_end]

    PRM1 = np.zeros(IRM.shape)
    PRM2 = np.zeros(IRM.shape)
    if mode == 'vertical':
        PRM1[::2] = IRM[::2]
        PRM2[:len(PRM2)//2] = IRM[:len(PRM2)//2]
    elif mode == 'horizontal':
        PRM1[:, ::2] = IRM[:, ::2]
        i_start, i_end = PRM2.shape[1]//4, PRM2.shape[1]*3//4
        PRM2[:, :i_start] = IRM[:, :i_start]
        PRM2[:, i_end:] = IRM[:, i_end:]
    elif mode == 'scaling':
        alpha = 0.5
        PRM1 = IRM*alpha
        PRM2 = 1 + alpha*(IRM-1)
    else:
        raise ValueError('Wrong mode!')

    MSE1 = ((IRM-PRM1)**2).mean()
    MSE2 = ((IRM-PRM2)**2).mean()

    with h5py.File(test_dataset_path, 'r') as f:
        mixture = f['mixtures'][mix_index].reshape(-1, 2)
        foreground = f['foregrounds'][mix_index].reshape(-1, 2)

    # scale signal
    scaler.fit(mixture)
    mixture = scaler.scale(mixture)
    foreground = scaler.scale(foreground)
    scaler.__init__(scaler.active)

    # apply filterbank
    mixture_filt = filterbank.filt(mixture)
    foreground_filt = filterbank.filt(foreground)

    # frame
    mixture_filt_framed = framer.frame(mixture_filt)
    MIX_I = (mixture_filt_framed**2).mean(axis=(1, 3))
    MIX_I = 10*np.log10(MIX_I)

    # apply predicted RM
    PRM1_ = wola(PRM1, trim=len(mixture_filt))[:, :, np.newaxis]
    PRM2_ = wola(PRM2, trim=len(mixture_filt))[:, :, np.newaxis]
    IRM_ = wola(IRM, trim=len(mixture_filt))[:, :, np.newaxis]
    mixture1 = filterbank.rfilt(mixture_filt*PRM1_)
    mixture2 = filterbank.rfilt(mixture_filt*PRM2_)
    mixture_best = filterbank.rfilt(mixture_filt*IRM_)
    mixture_ref = filterbank.rfilt(mixture_filt)
    foreground_ref = filterbank.rfilt(foreground_filt)

    # start MATLAB engine
    if matlab_engine is None:
        print('Starting MATLAB engine...')
        matlab_engine = matlab.engine.start_matlab()
        paths = [
            r'matlab',
            r'matlab\loizou',
            r'matlab\stft-framework\stft-framework\src\tools',
        ]
        for path in paths:
            matlab_engine.addpath(path, nargout=0)

    # remove noise-only parts
    npad = round(config.PRE.MIXTURES.PADDING*config.PRE.FS)
    mixture1 = mixture1[npad:-npad]
    mixture2 = mixture2[npad:-npad]
    mixture_ref = mixture_ref[npad:-npad]
    foreground_ref = foreground_ref[npad:-npad]
    mixture_best = mixture_best[npad:-npad]

    # flatten and convert to matlab float
    mixture1 = matlab.single(
        mixture1.sum(axis=1, keepdims=True).tolist())
    mixture2 = matlab.single(
        mixture2.sum(axis=1, keepdims=True).tolist())
    mixture_best = matlab.single(
        mixture_best.sum(axis=1, keepdims=True).tolist())
    mixture_ref = matlab.single(
        mixture_ref.sum(axis=1, keepdims=True).tolist())
    foreground_ref = matlab.single(
        foreground_ref.sum(axis=1, keepdims=True).tolist())

    # calculate PESQ
    pesq_before = matlab_engine.pesq(foreground_ref, mixture_ref,
                                     config.PRE.FS)
    pesq1 = matlab_engine.pesq(foreground_ref, mixture1,
                               config.PRE.FS)
    pesq2 = matlab_engine.pesq(foreground_ref, mixture2,
                               config.PRE.FS)
    pesq_best = matlab_engine.pesq(foreground_ref, mixture_best,
                                   config.PRE.FS)

    dpesq1 = pesq1 - pesq_before
    dpesq2 = pesq2 - pesq_before
    dpesq_best = pesq_best - pesq_before

    scaling_ = 0.525
    if layout == 'report':
        fig, axes = plt.subplots(4, 1, figsize=(15*scaling_, 12*scaling_))
    elif layout == 'ppt':
        fig, axes = plt.subplots(2, 2, figsize=(17*scaling_, 5.75*scaling_))
    else:
        raise ValueError('Wrong layout!')

    datas = [MIX_I, IRM, PRM1, PRM2]
    titles = [
        rf'Input mixture (PESQ: {pesq_before:.2f})',
        rf'IRM (\$\Delta\$PESQ: {dpesq_best:+.2f})',
        rf'PRM1 (\$\Delta\$PESQ: {dpesq1:+.2f}, MSE: {MSE1:.2f})',
        rf'PRM2 (\$\Delta\$PESQ: {dpesq2:+.2f}, MSE: {MSE2:.2f})',
    ]

    for i in range(4):
        ax = axes.T.ravel()[i]
        data = datas[i]
        title = titles[i]

        pos = ax.imshow(data.T, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)
        cbar = fig.colorbar(pos, cax=cax)
        if i == 0:
            cmin, cmax = cbar.ax.get_ylim()

        _, xmax = ax.get_xlim()
        xticklabels = np.arange(0, xmax/16e3*256, 0.5)
        xticks = xticklabels*16e3/256
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Time (s)')

        yticks = np.arange(0, 70, 15)
        ax.set_yticks(yticks)
        fc = filterbank.fc
        ax.set_yticklabels(fc[yticks].round().astype(int))
        ax.set_ylabel('Frequency (Hz)')

        if i in [1, 2, 3]:
            ax.get_images()[0].set_clim(0, 1)

        ax.set_title(title)
        ax.grid(False)

        pos = ax.title.get_position()
        ax.title.set_position([pos[0], pos[1]-0.04])
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis='both', which='major', pad=2, length=3)
        cbar.ax.tick_params(axis='both', which='major', pad=2, length=3)

    remove_patches(fig, axes.ravel())
    fig.subplots_adjust(
        # left=0.1,
        # right=0.9,
        # bottom=0.1,
        # top=0.9,
        # wspace=0.45,
        hspace=0.55,
    )

    if save_filepath is None:
        filename = f'mse_vs_pesq_{mode}_{layout}.svg'
        save_filepath = os.path.join(dirpath, filename)
    fig.savefig(save_filepath)
    plt.show()

    print(f'MSE1 {MSE1}')
    print(f'MSE2 {MSE2}')


if __name__ == '__main__':
    # example_specgrams()

    # experiment_regularization()
    # experiment_context()
    # experiment_layers()
    # experiment_features()
    # experiment_final_heatmap()
    # experiment_target_location(onlyreverb=False, testbig=True)
    # experiment_baseline()

    # experiment_dataset_size()
    # experiment_delta_delta_pesq()

    print('Starting MATLAB engine...')
    matlab_engine = matlab.engine.start_matlab()
    paths = [
        r'matlab',
        r'matlab\loizou',
        r'matlab\stft-framework\stft-framework\src\tools',
    ]
    for path in paths:
        matlab_engine.addpath(path, nargout=0)
    # example_enhancement(9, matlab_engine=matlab_engine, layout='ppt')
    # example_enhancement(9, matlab_engine=matlab_engine, layout='ppt',
    #                     foreback=True)
    example_enhancement_each_condition(matlab_engine=matlab_engine)
    # mse_vs_pesq_test(9, matlab_engine=matlab_engine, mode='vertical',
    #                  layout='ppt')
    # mse_vs_pesq_test(9, matlab_engine=matlab_engine, mode='horizontal',
    #                  layout='ppt')
    # mse_vs_pesq_test(9, matlab_engine=matlab_engine, mode='scaling',
    #                  layout='ppt')

    plt.show(block=True)
