import torch.nn as nn
import torch
import numpy as np

from .base import BreverBaseModel
from ..filters import STFT, MelFB
from ..features import FeatureExtractor


eps = np.finfo(float).eps


class DNN(BreverBaseModel):
    def __init__(
        self,
        fs=16000,
        features={'logfbe'},
        stacks=0,
        decimation=1,
        stft_frame_length=512,
        stft_hop_length=256,
        stft_window='hann',
        mel_filters=64,
        hidden_layers=[1024, 1024],
        dropout=0.2,
        batchnorm=False,
        batchnorm_momentum=0.1,
        normalization='static',
    ):
        super().__init__()
        self.stacks = stacks
        self.decimation = decimation
        self.stft = STFT(
            frame_length=stft_frame_length,
            hop_length=stft_hop_length,
            window=stft_window,
        )
        self.mel_fb = MelFB(
            n_filters=mel_filters,
            n_fft=stft_frame_length,
            fs=fs,
        )
        self.feature_extractor = FeatureExtractor(
            features=features,
            mel_fb=self.mel_fb,
            hop_length=stft_hop_length,
            fs=fs,
        )
        input_size = self.feature_extractor.n_features*(stacks+1)
        output_size = mel_filters
        self.dnn = _DNN(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=[1024, 1024],
            dropout=0.2,
            batchnorm=False,
            batchnorm_momentum=0.1,
        )
        if normalization == 'static':
            self.normalization = StaticNormalizer(input_size)
        elif normalization == 'cumulative':
            self.normalization = CumulativeNormalizer()
        else:
            raise ValueError('unrecognized normalization type, got '
                             f'{normalization}')

    def forward(self, x):
        return self.dnn(x)

    def pre_proc(self, data, target, return_stft_output=False):
        x = torch.stack([data, *target])  # (sources, channels, samples)
        mag, phase = self.stft.analyze(x)  # (sources, channels, bins, frames)
        mix_mag = mag[0, :, :, :]  # (channels, bins, frames)
        mix_phase = phase[0, :, :, :]  # (channels, bins, frames)
        fg_mag = mag[1, :, :, :]  # (channels, bins, frames)
        bg_mag = mag[2, :, :, :]  # (channels, bins, frames)
        # features
        data = self.feature_extractor((mix_mag, mix_phase))  # (feats, frames)
        data = self.stack(data)
        data = self.decimate(data)
        data = self.normalization(data)
        # labels
        target = self.irm(fg_mag, bg_mag)  # (labels, frames)
        target = self.decimate(target)
        if return_stft_output:
            return data, target, mix_mag, mix_phase
        else:
            return data, target

    def segment_to_item_length(self, segment_length):
        return self.stft.frame_count(segment_length)

    def enhance(self, x, return_mask=False):
        mag, phase = self.stft.analyze(x.unsqueeze(0))
        features = self.feature_extractor((mag.squeeze(0), phase.squeeze(0)))
        features = self.stack(features)
        features = self.normalization(features)
        mask = self.dnn(features.unsqueeze(0))
        mask_extrapolated = self.mel_fb.extrapolate(mask)
        mag *= mask_extrapolated
        x = self.stft.synthesize((mag, phase))[..., :x.shape[-1]]
        x, mask = x.squeeze(0), mask.squeeze(0)
        x = x.mean(dim=0)  # return monaural signal
        if return_mask:
            return x, mask
        else:
            return x

    def irm(self, fg_mag, bg_mag):
        # (sources, channels, bins, frames)
        fg_energy = fg_mag.pow(2).mean(0)  # (bins, frames)
        bg_energy = bg_mag.pow(2).mean(0)  # (bins, frames)
        fg_energy = self.mel_fb(fg_energy)
        bg_energy = self.mel_fb(bg_energy)
        irm = (1 + bg_energy/(fg_energy+eps)).pow(-0.5)
        return irm

    def stack(self, data):
        out = [data]
        for i in range(self.stacks):
            rolled = data.roll(i+1, -1)
            rolled[:, :i+1] = data[:, :1]
            out.append(rolled)
        return torch.cat(out)

    def decimate(self, data):
        return data[:, ::self.decimation]


class _DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[1024, 1024],
                 dropout=0.2, batchnorm=False, batchnorm_momentum=0.1):
        super().__init__()
        self.operations = nn.ModuleList()
        start_size = input_size
        for i in range(len(hidden_layers)):
            end_size = hidden_layers[i]
            self.operations.append(nn.Linear(start_size, end_size))
            if batchnorm:
                self.operations.append(
                    nn.BatchNorm1d(end_size, momentum=batchnorm_momentum)
                )
            self.operations.append(nn.ReLU())
            self.operations.append(nn.Dropout(dropout))
            start_size = end_size
        self.operations.append(nn.Linear(start_size, output_size))
        self.operations.append(nn.Sigmoid())

    def forward(self, x):
        x = x.transpose(1, 2)
        for operation in self.operations:
            x = operation(x)
        return x.transpose(1, 2)


class StaticNormalizer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        mean = torch.zeros((input_size, 1))
        std = torch.ones((input_size, 1))
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def set_statistics(self, mean, std):
        self.mean, self.std = mean, std

    def forward(self, x):
        return (x - self.mean)/self.std


class CumulativeNormalizer(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        batch_size, bins, frames = x.shape
        cum_sum = x.cumsum(-1)
        cum_pow_sum = x.pow(2).cumsum(-1)
        count = torch.arange(1, frames+1, device=x.device)
        count = count.reshape(1, 1, frames)
        cum_mean = cum_sum/count
        cum_var = cum_pow_sum/count - cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()
        return (x - cum_mean)/cum_std
