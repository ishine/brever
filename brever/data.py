import json
import os
import random
import tarfile
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

from .features import FeatureExtractor
from .filters import STFT, MelFB

eps = np.finfo(float).eps


class TarArchiveInterface:
    """
    Hats off to João F. Henriques for his tarfile PyTorch dataset available
    under a BSD-3-Clause License: https://github.com/jotaf98/simple-tar-dataset

    The difficulty comes from the fact that TarFile is not thread safe, so
    when using multiple workers, each worker should have its own file handle.
    """
    def __init__(self, archive):
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(archive)}
        self.archive = archive
        self.members = {m.name: m for m in self.tar_obj[worker].getmembers()}

    def get_file(self, name):
        # ensure a unique file handle per worker
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.archive)
        return self.tar_obj[worker].extractfile(self.members[name])


class BreverDataset(torch.utils.data.Dataset):
    """
    Base dataset for all datasets. Reads mixtures from a created dataset.
    Mixtures can be loaded as segments of fixed length or entirely by setting
    segment_length=0. When a fixed segment is used, the last samples in each
    mixture are dropped.

    Should be subclassed for post-processing by re-implementing the post_proc
    method.
    """
    def __init__(self, path, segment_length=4.0, overlap_length=0.0, fs=16e3,
                 components=['foreground', 'background'],
                 segment_strategy='pass', tar=True):
        self.path = path
        self.segment_length = round(segment_length*fs)
        self.overlap_length = round(overlap_length*fs)
        self.fs = fs
        self.components = components
        self.segment_strategy = segment_strategy
        if tar:
            self.archive = TarArchiveInterface(os.path.join(path, 'audio.tar'))
        else:
            self.archive = None
        self.segment_info = self.get_segment_info()
        self.preloaded_data = None
        self._item_lengths = None

    def get_file(self, name):
        if self.archive is None:
            file = open(os.path.join(self.path, name), 'rb')
        else:
            file = self.archive.get_file(name)
        return file

    def build_paths(self, mix_idx):
        mix_dir = 'audio'
        mix_path = os.path.join(mix_dir, f'{mix_idx:05d}_mixture.flac')
        comp_paths = []
        for comp in self.components:
            comp_path = os.path.join(mix_dir, f'{mix_idx:05d}_{comp}.flac')
            comp_paths.append(comp_path)
        return mix_path, comp_paths

    def get_segment_info(self):
        mix_lengths = self.get_mix_lengths()
        segment_info = []
        if self.segment_length == 0:
            for mix_idx, mix_length in enumerate(mix_lengths):
                segment_info.append((mix_idx, (0, mix_length)))
        else:
            for mix_idx, mix_length in enumerate(mix_lengths):
                self._add_segment_info(segment_info, mix_idx, mix_length)
        return segment_info

    def get_item_lengths(self):
        item_lengths = []
        for _, (start, end) in self.segment_info:
            item_lengths.append(self.segment_to_item_length(end-start))
        return item_lengths

    def segment_to_item_length(self, segment_length):
        raise NotImplementedError

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
            pass
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
            mix_file = self.get_file(mix_path)
            mix_metadata = torchaudio.info(mix_file)
            mix_length = mix_metadata.num_frames
            for comp_path in comp_paths:
                comp_file = self.get_file(mix_path)
                comp_metadata = torchaudio.info(comp_file)
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
        raise NotImplementedError

    def load_file(self, path):
        file = self.get_file(path)
        # x, _ = torchaudio.load(file)
        # torchaudio.load is BROKEN for file-like objects from FLAC files!
        # https://github.com/pytorch/audio/issues/2356
        # use soundfile instead
        x, fs = sf.read(file, dtype='float32')
        if x.ndim == 1:
            x = x.reshape(1, -1)
        else:
            x = x.T
        x = torch.from_numpy(x)
        return x, fs

    def load_segment(self, index):
        if not 0 <= index < len(self):
            raise IndexError
        mix_idx, (start, end) = self.segment_info[index]
        mix_path, comp_paths = self.build_paths(mix_idx)
        mix, _ = self.load_file(mix_path)
        if end > mix.shape[-1]:
            if self.segment_strategy != 'pad':
                raise ValueError('attempting to load a segment outside of '
                                 'mixture range but segment strategy is not '
                                 f'"pad", got "{self.segment_strategy}"')
            mix = F.pad(mix, (0, end - mix.shape[-1]))
        mix = mix[:, start:end]
        components = []
        for comp_path in comp_paths:
            component, _ = self.load_file(comp_path)
            component = component[:, start:end]
            components.append(component)
        return mix, torch.stack(components)

    def __len__(self):
        return len(self.segment_info)

    @property
    def item_lengths(self):
        if self._item_lengths is None:
            self._item_lengths = self.get_item_lengths()
        return self._item_lengths

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
            fs=self.fs,
        )
        self.feature_extractor = FeatureExtractor(
            features=features,
            mel_fb=self.mel_fb,
            hop_length=stft_hop_length,
            fs=self.fs,
        )

    def segment_to_item_length(self, item_length):
        return self.stft.frame_count(item_length)

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

    def segment_to_item_length(self, item_length):
        return item_length

    def post_proc(self, data, target):
        data = data.mean(axis=-2)
        target = target.mean(axis=-2)
        return data, target


class BreverBatchSampler(torch.utils.data.Sampler):
    """
    Base class for all samplers.
    """
    def __init__(self, dataset, drop_last=False, shuffle=True, seed=0,
                 sort=False):
        self.__pre_init__(dataset, drop_last, shuffle, seed, sort)
        self.generate_batches()

    def __pre_init__(self, dataset, drop_last, shuffle, seed, sort):
        self.dataset = dataset
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.sort = sort
        self._epoch = 0
        self._previous_epoch = -1
        self._item_lengths = self.get_item_lengths()

    def generate_batches(self):
        indices = self._generate_indices()
        self._generate_batches(indices)

    def _generate_indices(self):
        if self.sort:
            if self.shuffle:
                # sort by length but randomize items of same length
                randomizer = random.Random(self.seed + self._epoch)
                lengths = sorted(self._item_lengths,
                                 key=lambda x: (x[1], randomizer.random()))
            else:
                lengths = sorted(self._item_lengths, key=lambda x: x[1])
            indices = [idx for idx, length in lengths]
        else:
            indices = list(range(len(self._item_lengths)))
            if self.shuffle:
                randomizer = random.Random(self.seed + self._epoch)
                randomizer.shuffle(indices)
        return indices

    def _generate_batches(self, indices):
        raise NotImplementedError

    def shuffle_batches(self):
        randomizer = random.Random(self.seed + self._epoch)
        randomizer.shuffle(self.batches)

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

    def segment_to_item_length(self, segment_length):
        if isinstance(self.dataset, torch.utils.data.Subset):
            dataset = self.dataset.dataset
        return dataset.segment_to_item_length(segment_length)

    def calc_batch_stats(self):
        batch_sizes = []
        pad_amounts = []
        for batch in self.batches:
            batch_lengths = [length for idx, length in batch]
            max_length = max(batch_lengths)
            batch_sizes.append(len(batch)*max_length)
            pad_amounts.append(
                sum(max_length - length for length in batch_lengths)
            )
        return batch_sizes, pad_amounts

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        if self.shuffle:
            if self._epoch == self._previous_epoch:
                raise ValueError('the set_epoch method must be called before '
                                 'iterating over the dataloader in order to '
                                 'regenerate the batches with the correct '
                                 'seed')
            self.generate_batches()
            self.shuffle_batches()
            self._previous_epoch = self._epoch
        for batch in self.batches:
            yield [idx for idx, length in batch]

    def __len__(self):
        return len(self.batches)


class SimpleBatchSampler(BreverBatchSampler):
    """
    Simplest batch sampler possible. Batches have a fixed number of items.
    Iterates over items and appends to current batch if it fits, otherwise
    appends to new batch. This means batches have random total size and
    padding.
    """
    def __init__(self, dataset, items_per_batch, drop_last=False, shuffle=True,
                 seed=0):
        super().__pre_init__(dataset, drop_last, shuffle, seed, sort=False)
        self.items_per_batch = items_per_batch
        self.generate_batches()

    def _generate_batches(self, indices):
        self.batches = []
        batch = []
        for i in indices:
            item_idx, item_length = self._item_lengths[i]
            batch.append((item_idx, item_length))
            if len(batch) == self.items_per_batch:
                self.batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            self.batches.append(batch)


class DynamicSimpleBatchSampler(BreverBatchSampler):
    """
    Similar to SimpleBatchSampler but with a dynamic number of items per batch.
    Items are added to the current batch until the total batch size exceeds a
    limit. This can subtantially reduce the batch size variability and the
    total number of batches, but increases padding.
    """
    def __init__(self, dataset, max_batch_size, drop_last=False, shuffle=True,
                 seed=0):
        super().__pre_init__(dataset, drop_last, shuffle, seed, sort=False)
        max_batch_size = self.segment_to_item_length(max_batch_size)
        self.max_batch_size = max_batch_size
        self.generate_batches()

    def _generate_batches(self, indices):
        self.batches = []
        batch = []
        batch_width = 0
        for i in indices:
            item_idx, item_length = self._item_lengths[i]
            if item_length > self.max_batch_size:
                raise ValueError('found an item that is longer than the '
                                 'maximum batch size')
            if (len(batch)+1)*max(item_length, batch_width) \
                    <= self.max_batch_size:
                batch.append((item_idx, item_length))
                batch_width = max(item_length, batch_width)
            else:
                self.batches.append(batch)
                batch = []
                batch.append((item_idx, item_length))
                batch_width = item_length
        if len(batch) > 0 and not self.drop_last:
            self.batches.append(batch)


class SortedBatchSampler(SimpleBatchSampler):
    """
    Sorts all items by length before making batches with a fixed number of
    items. This optimally minimizes the amount of padding, but batches have
    highly variable size. Namely the last batches can be very small if
    the dataset contains very short items. Also items within a batch are not
    random.
    """
    def __init__(self, dataset, items_per_batch, drop_last=False, shuffle=True,
                 seed=0):
        super().__pre_init__(dataset, drop_last, shuffle, seed, sort=True)
        self.items_per_batch = items_per_batch
        self.generate_batches()


class DynamicSortedBatchSampler(DynamicSimpleBatchSampler):
    """
    Similar to SortedBatchSampler but with a dynamic number of items per batch.
    After sorting, items are added to the current batch until the total batch
    size exceeds a limit. This can subtantially reduce the batch size
    variability and the total number of batches, but increases padding. Items
    within a batch are also not random because of sorting.
    """
    def __init__(self, dataset, max_batch_size, drop_last=False,
                 shuffle=True, seed=0):
        super().__pre_init__(dataset, drop_last, shuffle, seed, sort=True)
        max_batch_size = self.segment_to_item_length(max_batch_size)
        self.max_batch_size = max_batch_size
        self.generate_batches()


class BucketBatchSampler(BreverBatchSampler):
    """
    Items of similar length are grouped into buckets. Batches are formed with
    items from the same bucket. This attempts to minize both the batch size
    variability and the amount of padding while keeping some randomness.

    Inspired from code by Speechbrain under Apache-2.0 License:
    https://github.com/speechbrain/speechbrain/blob/b5d2836e3d0eabb541c5bdbca16fb00c49cb62a3/speechbrain/dataio/sampler.py#L305
    """
    def __init__(self, dataset, max_batch_size, max_item_length,
                 num_buckets=10, drop_last=False, shuffle=True, seed=0):
        super().__pre_init__(dataset, drop_last, shuffle, seed, sort=False)
        if max_batch_size < max_item_length:
            raise ValueError('cannot have max_batch_size < max_item_length, '
                             f'got max_batch_size={max_batch_size} '
                             f'and max_item_length={max_item_length}')
        max_batch_size = self.segment_to_item_length(max_batch_size)
        max_item_length = self.segment_to_item_length(max_item_length)
        self.max_batch_size = max_batch_size
        self.max_item_length = max_item_length
        self.num_buckets = num_buckets

        self.right_bucket_limits = np.linspace(
            max_item_length/num_buckets, max_item_length, num_buckets,
        )
        self.bucket_batch_lengths = max_batch_size//self.right_bucket_limits

        self.generate_batches()

    def _generate_batches(self, indices):
        self.batches = []
        bucket_batches = [[] for _ in range(self.num_buckets)]
        for i in indices:
            item_idx, item_length = self._item_lengths[i]
            bucket_idx = np.searchsorted(
                self.right_bucket_limits, item_length,
            )
            if bucket_idx == self.num_buckets:
                if item_length == self.max_item_length:
                    bucket_idx -= 1
                else:
                    raise ValueError('found an item that is longer than the '
                                     'maximum item length')
            bucket_batches[bucket_idx].append((item_idx, item_length))
            if len(bucket_batches[bucket_idx]) \
                    == self.bucket_batch_lengths[bucket_idx]:
                self.batches.append(bucket_batches[bucket_idx])
                bucket_batches[bucket_idx] = []
            elif len(bucket_batches[bucket_idx]) \
                    > self.bucket_batch_lengths[bucket_idx]:
                raise ValueError('bucket maximum number of items exceeded')
        if not self.drop_last:
            for batch in bucket_batches:
                if len(batch) > 0:
                    self.batches.append(batch)


class BreverDataLoader(torch.utils.data.DataLoader):
    """
    Implements the collating function to form batches from mixtures of
    different length.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def set_epoch(self, epoch):
        self.batch_sampler.set_epoch(epoch)

    def _collate_fn(self, batch):
        lengths = list(x.shape[-1] for x, _ in batch)
        max_length = max(lengths)
        batch_x, batch_y = [], []
        for x, y in batch:
            assert x.shape[-1] == y.shape[-1]
            padding = max_length - x.shape[-1]
            batch_x.append(F.pad(x, (0, padding)))
            batch_y.append(F.pad(y, (0, padding)))
        return torch.stack(batch_x), torch.stack(batch_y), lengths


def initialize_dataset(config, cuda):
    # initialize dataset
    logging.info('Initializing dataset')
    print(config.TRAINING.PATH)
    if config.ARCH == 'dnn':
        dataset = DNNDataset(
            path=config.TRAINING.PATH,
            segment_length=config.TRAINING.SEGMENT_LENGTH,
            fs=config.FS,
            features=config.MODEL.FEATURES,
            stacks=config.MODEL.STACKS,
            decimation=config.MODEL.DECIMATION,
            stft_frame_length=config.MODEL.STFT.FRAME_LENGTH,
            stft_hop_length=config.MODEL.STFT.HOP_LENGTH,
            stft_window=config.MODEL.STFT.WINDOW,
            mel_filters=config.MODEL.MEL_FILTERS,
        )
    elif config.ARCH == 'convtasnet':
        dataset = ConvTasNetDataset(
            path=config.TRAINING.PATH,
            segment_length=config.TRAINING.SEGMENT_LENGTH,
            fs=config.FS,
            components=config.MODEL.SOURCES,
        )
    else:
        raise ValueError(f'wrong model architecture, got {config.ARCH}')

    # preload data
    if config.TRAINING.PRELOAD:
        logging.info('Preloading data')
        dataset.preload(cuda)

    # train val split
    val_length = int(len(dataset)*config.TRAINING.VAL_SIZE)
    train_length = len(dataset) - val_length
    train_split, val_split = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )

    return dataset, train_split, val_split
