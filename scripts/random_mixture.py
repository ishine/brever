import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np

from brever.config import defaults
import brever.display as bplot
from brever import RandomMixtureMaker, Filterbank, Framer


def main():
    # seed for reproducibility
    random.seed(12)

    # read default config
    config = defaults()

    # init random mixture maker
    rmm = RandomMixtureMaker(
        fs=config.PRE.FS,
        rooms=config.PRE.MIX.RANDOM.ROOMS,
        speakers=config.PRE.MIX.RANDOM.TARGET.SPEAKERS,
        target_snr_dist_name=config.PRE.MIX.RANDOM.TARGET.SNR.DISTNAME,
        target_snr_dist_args=config.PRE.MIX.RANDOM.TARGET.SNR.DISTARGS,
        target_angle_min=config.PRE.MIX.RANDOM.TARGET.ANGLE.MIN,
        target_angle_max=config.PRE.MIX.RANDOM.TARGET.ANGLE.MAX,
        dir_noise_nums=range(
            config.PRE.MIX.RANDOM.SOURCES.NUMBER.MIN,
            config.PRE.MIX.RANDOM.SOURCES.NUMBER.MAX + 1,
        ),
        dir_noise_types=config.PRE.MIX.RANDOM.SOURCES.TYPES,
        dir_noise_snrs=range(
            config.PRE.MIX.RANDOM.SOURCES.SNR.MIN,
            config.PRE.MIX.RANDOM.SOURCES.SNR.MAX + 1,
        ),
        dir_noise_angle_min=config.PRE.MIX.RANDOM.SOURCES.ANGLE.MIN,
        dir_noise_angle_max=config.PRE.MIX.RANDOM.SOURCES.ANGLE.MAX,
        diffuse_noise_on=config.PRE.MIX.DIFFUSE.ON,
        diffuse_noise_color=config.PRE.MIX.DIFFUSE.COLOR,
        diffuse_noise_ltas_eq=config.PRE.MIX.DIFFUSE.LTASEQ,
        mixture_pad=config.PRE.MIX.PADDING,
        mixture_rb=config.PRE.MIX.REFLECTIONBOUNDARY,
        mixture_rms_jitter_on=config.PRE.MIX.RANDOM.RMSDB.ON,
        mixture_rms_jitters=range(
            config.PRE.MIX.RANDOM.RMSDB.MIN,
            config.PRE.MIX.RANDOM.RMSDB.MAX + 1,
        ),
        filelims_dir_noise=config.PRE.MIX.FILELIMITS.NOISE,
        filelims_target=config.PRE.MIX.FILELIMITS.TARGET,
        filelims_room=config.PRE.MIX.FILELIMITS.ROOM,
        decay_on=config.PRE.MIX.DECAY.ON,
        decay_color=config.PRE.MIX.DECAY.COLOR,
        decay_rt60_dist_name=config.DATASET.DECAY.RT60.DISTNAME,
        decay_rt60_dist_args=config.DATASET.DECAY.RT60.DISTARGS,
        decay_drr_dist_name=config.DATASET.DECAY.DRR.DISTNAME,
        decay_drr_dist_args=config.DATASET.DECAY.DRR.DISTARGS,
        decay_delay_dist_name=config.DATASET.DECAY.DELAY.DISTNAME,
        decay_delay_dist_args=config.DATASET.DECAY.DELAY.DISTARGS,
        seed_on=config.PRE.SEED.ON,
        seed_value=config.PRE.SEED.VALUE,
        uniform_tmr=config.PRE.MIX.RANDOM.UNIFORMTMR,
    )

    # init filterbank
    filterbank = Filterbank(
        kind=config.PRE.FILTERBANK.KIND,
        n_filters=config.PRE.FILTERBANK.NFILTERS,
        f_min=config.PRE.FILTERBANK.FMIN,
        f_max=config.PRE.FILTERBANK.FMAX,
        fs=config.PRE.FS,
        order=config.PRE.FILTERBANK.ORDER,
    )
    framer = Framer(
        frame_length=config.PRE.FRAMER.FRAMELENGTH,
        hop_length=config.PRE.FRAMER.HOPLENGTH,
        window=config.PRE.FRAMER.WINDOW,
        center=config.PRE.FRAMER.CENTER,
    )

    # make a random mixture and print metadata
    for i in range(100):
        mixObj, metadata = rmm.make()
    print(json.dumps(metadata, indent=4))

    # plot waveform and spectrogram for each component
    attributes = ['mixture', 'target', 'noise']
    fig, axes = plt.subplots(len(attributes), 2)
    for i, attribute in enumerate(attributes):
        x = getattr(mixObj, attribute)

        # average across left and right channels
        x_mono = x.mean(axis=1)

        # plot waveform
        bplot.plot_waveform(
            x_mono,
            ax=axes[i, 0],
            fs=rmm.fs,
            set_kw={'title': attribute},
        )

        # filter
        x_filtered = filterbank.filt(x_mono)

        # frame
        x_framed = framer.frame(x_filtered)

        # average energy in each frame
        X = (x_framed**2).mean(axis=1)

        # convert to dB
        X_dB = 10*np.log10(X + np.finfo(float).eps)

        # plot spectrogram
        bplot.plot_spectrogram(
            X_dB,
            ax=axes[i, 1],
            fs=rmm.fs,
            hop_length=framer.hop_length,
            f=filterbank.fc,
            set_kw={'title': attribute},
        )

    # match limits
    bplot.share_ylim(axes[:, 0], center_zero=True)
    bplot.share_clim(axes[:, 1])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create and plot a random '
                                                 'mixture')
    args = parser.parse_args()
    main()
