import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .features import FeatureExtractor
from .filters import STFT, MelFB

eps = np.finfo(float).eps


class BreverDataset(torch.utils.data.Dataset):
    """
    Base dataset for all datasets. Reads mixtures from a created dataset.
    Mixtures can be loaded as segments of fixed length or entirely by setting
    segment_length=0. When a fixed segment is used, the last samples in each
    mixture are dropped.

    Should be subclassed for post-processing by re-implementing the post_proc
    method.
    """
    def __init__(self, path, segment_length=0, overlap_length=0, fs=16e3,
                 components=['foreground', 'background'],
                 segment_strategy='drop'):
        self.path = path
        self.segment_length = round(segment_length*fs)
        self.overlap_length = round(overlap_length*fs)
        self.components = components
        self.segment_strategy = segment_strategy
        self.segment_info = self.get_segment_info()
        self.preloaded_data = None

    def build_paths(self, mix_idx):
        mix_dir = os.path.join(self.path, 'audio')
        mix_path = os.path.join(mix_dir, f'{mix_idx:05d}_mixture.flac')
        comp_paths = []
        for comp in self.components:
            comp_path = os.path.join(mix_dir, f'{mix_idx:05d}_{comp}.flac')
            comp_paths.append(comp_path)
        return mix_path, comp_paths

    def get_segment_info(self):
        mix_lengths = self.get_mix_lengths()
        segment_info = []
        self._pad_amount = 0
        self._drop_amount = 0
        if self.segment_length == 0:
            for mix_idx, mix_length in enumerate(mix_lengths):
                segment_info.append((mix_idx, (0, mix_length)))
        else:
            for mix_idx, mix_length in enumerate(mix_lengths):
                self._add_segment_info(segment_info, mix_idx, mix_length)
        return segment_info

    def _add_segment_info(self, segment_info, mix_idx, mix_length):
        # shift length
        hop_length = self.segment_length - self.overlap_length
        # number of segments in mixture
        n_segments = (mix_length - self.segment_length)//hop_length + 1
        # build segments
        for segment_idx in range(n_segments):
            start = segment_idx*hop_length
            end = start + self.segment_length
            segment_info.append((mix_idx, (start, end)))
        # if segment_length > mix_length then we never entered the loop
        # we need to assign the last index of the last segment to 0
        # such that handling of the remaining samples does not fail
        if n_segments <= 0:
            end = 0
        if self.segment_strategy == 'drop':
            self._drop_amount += mix_length - end
        elif self.segment_strategy == 'pass':
            if end != mix_length:
                segment_idx = n_segments
                start = segment_idx*hop_length
                end = mix_length
                segment_info.append((mix_idx, (start, end)))
        elif self.segment_strategy == 'pad':
            if end != mix_length:
                segment_idx = n_segments
                start = segment_idx*hop_length
                end = start + self.segment_length
                segment_info.append((mix_idx, (start, end)))
                self._pad_amount = end - mix_length
        elif self.segment_strategy == 'overlap':
            if end != mix_length:
                start = mix_length - self.segment_length
                end = mix_length
                segment_info.append((mix_idx, (start, end)))
        else:
            raise ValueError('wrong segment strategy, got '
                             f'{self.segment_strategy}')

    def get_mix_lengths(self):
        json_path = os.path.join(self.path, 'mixture_info.json')
        with open(json_path) as f:
            json_data = json.load(f)
        n_mix = len(json_data)
        mix_lengths = []
        for i in range(n_mix):
            mix_path, comp_paths = self.build_paths(i)
            mix_metadata = torchaudio.info(mix_path)
            mix_length = mix_metadata.num_frames
            for comp_path in comp_paths:
                comp_metadata = torchaudio.info(comp_path)
                comp_length = comp_metadata.num_frames
                assert mix_length == comp_length
            mix_lengths.append(mix_length)
        return mix_lengths

    def __getitem__(self, index):
        if self.preloaded_data is not None:
            data, target = self.preloaded_data[index]
        else:
            data, target = self.load_segment(index)
            data, target = self.post_proc(data, target)
        return data, target

    def post_proc(self, data, target):
        return data, target

    def load_segment(self, index):
        if not 0 <= index < len(self):
            raise IndexError
        mix_idx, (start, end) = self.segment_info[index]
        mix_path, comp_paths = self.build_paths(mix_idx)
        mix, _ = torchaudio.load(mix_path)
        if end > mix.shape[-1]:
            if self.segment_strategy != 'pad':
                raise ValueError('attempting to load a segment outside of '
                                 'mixture range but segment strategy is not '
                                 f'"pad", got "{self.segment_strategy}"')
            mix = F.pad(mix, (0, end - mix.shape[-1]))
        mix = mix[:, start:end]
        components = []
        for comp_path in comp_paths:
            component, _ = torchaudio.load(comp_path)
            component = component[:, start:end]
            components.append(component)
        return mix, torch.stack(components)

    def __len__(self):
        return len(self.segment_info)

    @property
    def item_lengths(self):
        return [end-start for _, (start, end) in self.segment_info]

    def preload(self, cuda):
        preloaded_data = []
        for i in range(len(self)):
            data, target = self[i]
            if cuda:
                data, target = data.cuda(), target.cuda()
            preloaded_data.append((data, target))
        # set the attribute only at the end, otherwise __getitem__ will attempt
        # to access it insite the loop
        self.preloaded_data = preloaded_data


class BreverBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that minimizes the amount of padding required to collate
    mixtures of different length. This is done by sorting the mixtures by
    length. So within a batch the order is deterministic, but batches can
    be shuffled. The dataset must have an item_lengths attribute containing
    the list of each item length such that the batch list can be created
    without manually iterating across the dataset.
    """
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batches = self.generate_batches()

    def generate_batches(self):
        lengths = self.get_item_lengths()
        lengths = sorted(lengths, key=lambda x: x[1])
        batch_list = []
        batch = []
        _batch_lengths = []
        self._pad_amount = 0
        for idx, length in lengths:
            batch.append(idx)
            _batch_lengths.append(length)
            if len(batch) == self.batch_size:
                batch_list.append(batch)
                self._pad_amount += self._calc_batch_pad_amount(_batch_lengths)
                batch = []
                _batch_lengths = []
        if len(batch) > 0 and not self.drop_last:
            batch_list.append(batch)
            self._pad_amount += self._calc_batch_pad_amount(_batch_lengths)
        return batch_list

    def _calc_batch_pad_amount(self, _batch_lengths):
        max_length = max(_batch_lengths)
        return sum(max_length - length for length in _batch_lengths)

    def get_item_lengths(self):
        if isinstance(self.dataset, torch.utils.data.Subset):
            dataset = self.dataset.dataset
            indices = self.dataset.indices
            lengths = []
            for i, index in enumerate(indices):
                lengths.append((i, dataset.item_lengths[index]))
        else:
            lengths = list(enumerate(self.dataset.item_lengths))
        return lengths

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for i in self.batches:
            yield i

    def __len__(self):
        return len(self.batches)


class BreverDataLoader(torch.utils.data.DataLoader):
    """
    Implements the collating function to form batches from mixtures of
    different length.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        max_length = max(x.shape[-1] for x, _ in batch)
        batch_x, batch_y = [], []
        for x, y in batch:
            assert x.shape[-1] == y.shape[-1]
            batch_x.append(F.pad(x, (0, max_length - x.shape[-1])))
            batch_y.append(F.pad(y, (0, max_length - y.shape[-1])))
        return torch.stack(batch_x), torch.stack(batch_y)


class DNNDataset(BreverDataset):
    def __init__(
        self,
        path,
        features={'logfbe'},
        stacks=0,
        decimation=1,
        stft_frame_length=512,
        stft_hop_length=256,
        stft_window='hann',
        mel_filters=64,
        fs=16e3,
        **kwargs,
    ):
        super().__init__(path, components=['foreground', 'background'],
                         **kwargs)
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

    def post_proc(self, data, target, return_stft_output=False):
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
        # labels
        target = self.irm(fg_mag, bg_mag)  # (labels, frames)
        target = self.decimate(target)
        if return_stft_output:
            return data, target, mix_mag, mix_phase
        else:
            return data, target

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

    @property
    def item_lengths(self):
        return [self.stft.frame_count(n) for n in super().item_lengths]

    @property
    def n_features(self):
        data, _ = self[0]
        return data.shape[0]

    @property
    def n_labels(self):
        _, label = self[0]
        return label.shape[0]

    def get_statistics(self):
        mean, pow_mean = 0, 0
        for i in range(len(self)):
            data, _ = self[i]
            mean += data.mean(1, keepdim=True)
            pow_mean += data.pow(2).mean(1, keepdim=True)
        mean /= len(self)
        pow_mean /= len(self)
        var = pow_mean - mean.pow(2)
        std = var.sqrt()
        return mean, std


class ConvTasNetDataset(BreverDataset):
    def __init__(
        self,
        path,
        **kwargs,
    ):
        super().__init__(path, **kwargs)

    def post_proc(self, data, target):
        data = data.mean(axis=-2)
        target = target.mean(axis=-2)
        return data, target
