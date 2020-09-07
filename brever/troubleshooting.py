import os
import json
import pickle

import yaml
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import IPython.display as ipyd
import matlab
import matlab.engine

from .config import defaults
from .utils import wola
from .modelmanagement import get_feature_indices, get_file_indices
from .pytorchtools import TensorStandardizer, Feedforward, H5Dataset


def troubleshoot(model_dir, test_dataset_dir, n_mixtures):
    test_dataset_path = os.path.join(test_dataset_dir, 'dataset.hdf5')

    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    config = defaults()
    config.update(data)

    print(json.dumps(data, indent=4))

    if config.POST.GLOBALSTANDARDIZATION:
        stat_path = os.path.join(model_dir, 'statistics.npy')
        mean, std = np.load(stat_path)

    feature_indices = get_feature_indices(test_dataset_dir,
                                          config.POST.FEATURES)
    file_indices = get_file_indices(test_dataset_dir)
    test_dataset = H5Dataset(
        filepath=test_dataset_path,
        load=config.POST.LOAD,
        stack=config.POST.STACK,
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
    state_file = os.path.join(model_dir, 'checkpoint.pt')
    model.load_state_dict(torch.load(state_file))

    mixtures_info_filepath = os.path.join(test_dataset_dir,
                                          'mixture_info.json')
    with open(mixtures_info_filepath) as f:
        mixtures_info = json.load(f)

    pipes_file = os.path.join(test_dataset_dir, 'pipes.pkl')
    with open(pipes_file, 'rb') as f:
        pipes = pickle.load(f)
    scaler = pipes['scaler']
    filterbank = pipes['filterbank']
    framer = pipes['framer']

    eng = matlab.engine.start_matlab()
    eng.addpath('..\\matlab\\loizou', nargout=0)

    n_features = test_dataset.n_features//(config.POST.STACK+1)
    with h5py.File(test_dataset_path, 'r') as f:
        for k in range(n_mixtures):
            ipyd.display(ipyd.Markdown(f'**MIXTURE {k+1}:**'))
            print(json.dumps(mixtures_info[k], indent=4))

            mixture = f['mixtures'][k].reshape(-1, 2)
            foreground = f['foregrounds'][k].reshape(-1, 2)
            ipyd.display(ipyd.Audio(mixture.T, rate=16e3, normalize=True))

            scaler.fit(mixture)
            mixture = scaler.scale(mixture)
            foreground = scaler.scale(foreground)
            scaler.__init__(scaler.active)

            mixture_filt = filterbank.filt(mixture)
            foreground_filt = filterbank.filt(foreground)

            energy = framer.frame(mixture_filt)
            energy = np.mean(energy**2, axis=1)
            energy = 10*np.log10(energy + 1e-10)

            i_start, i_end = file_indices[k]
            features, labels = test_dataset[i_start:i_end]
            model.eval()
            with torch.no_grad():
                prediction = model(features)
            features_normalized = features[:, :n_features]
            features_raw = ((features_normalized*std[:n_features])
                            + mean[:n_features])

            prediction_extrapolated = wola(prediction, trim=len(mixture_filt))
            prediction_extrapolated = prediction_extrapolated[:, :, np.newaxis]
            labels_extrapolated = wola(labels, trim=len(mixture_filt))
            labels_extrapolated = labels_extrapolated[:, :, np.newaxis]
            mixture_enhanced = filterbank.rfilt((mixture_filt
                                                 * prediction_extrapolated))
            best_mixture_enhanced = filterbank.rfilt((mixture_filt
                                                      * labels_extrapolated))
            mixture_ref = filterbank.rfilt(mixture_filt)
            foreground_ref = filterbank.rfilt(foreground_filt)

            height_ratios = [1, 1, 1, n_features//64, n_features//64, 1, 1]
            gridspec_kw = {'height_ratios': height_ratios}
            fig, axes = plt.subplots(7, 1, figsize=(10, 16),
                                     gridspec_kw=gridspec_kw)
            ax = axes[0]
            ax.plot(mixture)
            for i, (x, title) in enumerate(zip([
                        energy[:, :, 0],
                        energy[:, :, 1],
                        features_raw,
                        features_normalized,
                        labels,
                        prediction,
                    ], [
                        'left',
                        'right',
                        'features_raw',
                        'features_normalized',
                        'labels',
                        'prediction',
                    ])):
                ax = axes[i+1]
                pos = ax.imshow(x.T, aspect='auto', origin='lower')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size=.1, pad=0.1)
                fig.colorbar(pos, cax=cax)
                ax.set_title(title)
            fig.tight_layout()
            plt.show()

            ipyd.display(ipyd.Audio(mixture_enhanced.T, rate=16e3,
                                    normalize=True))

            npad = round(config.PRE.MIXTURES.PADDING*config.PRE.FS)
            mixture_enhanced = mixture_enhanced[npad:-npad]
            best_mixture_enhanced = best_mixture_enhanced[npad:-npad]
            mixture_ref = mixture_ref[npad:-npad]
            foreground_ref = foreground_ref[npad:-npad]

            mixture_enhanced = matlab.single(
                mixture_enhanced.sum(axis=1, keepdims=True).tolist())
            best_mixture_enhanced = matlab.single(
                best_mixture_enhanced.sum(axis=1, keepdims=True).tolist())
            mixture_ref = matlab.single(
                mixture_ref.sum(axis=1, keepdims=True).tolist())
            foreground_ref = matlab.single(
                foreground_ref.sum(axis=1, keepdims=True).tolist())

            pesq_before = eng.pesq(foreground_ref, mixture_ref, config.PRE.FS)
            pesq_after = eng.pesq(foreground_ref, mixture_enhanced,
                                  config.PRE.FS)
            delta_pesq = pesq_after - pesq_before
            print(f'DeltaPESQ: {delta_pesq}')

            pesq_after = eng.pesq(foreground_ref, best_mixture_enhanced,
                                  config.PRE.FS)
            delta_pesq = pesq_after - pesq_before
            print(f'Best achievable DeltaPESQ: {delta_pesq}')

            print('\n\n')
