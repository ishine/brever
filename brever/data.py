import json
import logging
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import h5py

from .features import FeatureExtractor
from .utils import dct
from .filters import Filterbank

eps = np.finfo(float).eps


def get_mean_and_std(dataset, dataloader, uniform_stats_features):
    if dataset.load:
        data, _ = dataset[:]
        mean = data.mean(axis=0)
        std = data.std(axis=0)
    else:
        mean = 0
        for data, _ in dataloader:
            mean += data.mean(axis=0)
        mean /= len(dataloader)
        var = 0
        for data, _ in dataloader:
            var += ((data - mean)**2).mean(axis=0)
        var /= len(dataloader)
        std = var**0.5
        mean, std = mean.numpy(), std.numpy()
    for feature in uniform_stats_features:
        if feature not in dataset.features:
            raise ValueError(f'Uniform standardization feature "{feature}"" '
                             'is not in the list of selected features: '
                             f'{dataset.features}')
    mean, std = unify_stats(mean, std, dataset, uniform_stats_features)
    return mean, std


def get_files_mean_and_std(dataset, uniform_stats_features):
    means = []
    stds = []
    for i_start, i_end in dataset.file_indices:
        i_start = i_start // dataset.decimation
        i_end = i_end // dataset.decimation
        features, _ = dataset[i_start:i_end]
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        mean, std = unify_stats(mean, std, dataset, uniform_stats_features)
        means.append(mean)
        stds.append(std)
    return means, stds


def unify_stats(mean, std, dataset, uniform_stats_features):
    i_start = 0
    for feature, indices in zip(dataset.features, dataset.feature_indices):
        i_start_dataset, i_end_dataset = indices
        # i_start is the starting feature index in the stats vector
        # i_start_dataset is the starting feature index in the dataset
        i_end = i_start + i_end_dataset - i_start_dataset
        if feature in uniform_stats_features:
            feature_mean = mean[i_start:i_end].mean()
            feature_std = np.sqrt(
                np.mean(std[i_start:i_end]**2)
                + np.mean(mean[i_start:i_end]**2)
                - feature_mean**2
            )
            mean[i_start:i_end] = feature_mean
            std[i_start:i_end] = feature_std
        i_start = i_end
    return mean, std


class TensorStandardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean)/self.std


class StateTensorStandardizer:
    def __init__(self, means, stds, state=0):
        self.means = means
        self.stds = stds
        self.state = 0

    def set_state(self, state):
        self.state = state

    def __call__(self, data):
        mean = self.means[self.state]
        std = self.stds[self.state]
        return (data - mean)/std


class ResursiveTensorStandardizer:
    def __init__(self, mean=0, std=1, momentum=0.99):
        self.init_mean = mean
        self.init_std = std
        self.mean = mean
        self.std = std
        self.momentum = momentum

    def reset(self):
        self.mean = self.init_mean
        self.std = self.init_std

    def __call__(self, data):
        self.mean = self.mean*self.momentum + (1-self.momentum)*data
        self.std = (self.momentum*self.std**2 +
                    (1-self.momentum)*(data-self.mean)**2)**0.5
        return (data - self.mean)/self.std


class H5Dataset(torch.utils.data.Dataset):
    '''
    Custom dataset class that loads HDF5 files created by create_dataset.py

    Performs decimation, context stacking and feature selection

    Inspired from an extensive discussion on the PyTorch forums about how to
    efficiently implement HDF5 dataset classes:
    https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16,
    '''
    def __init__(self, dirpath, features=None, labels=None, load=False,
                 transform=None, stack=0, decimation=1, dct_toggle=False,
                 n_dct=5, prestack=False):
        if features is not None:
            features = sorted(features)
        if labels is not None:
            labels = sorted(labels)
        self.dirpath = dirpath
        self.features = features
        self.labels = labels
        self.load = load
        self.stack = stack
        self.decimation = decimation
        self.dct_toggle = dct_toggle
        self.n_dct = n_dct
        self.prestack = prestack
        self.filepath = os.path.join(dirpath, 'dataset.hdf5')
        self._prestacked = False
        self.datasets = None
        self.file_nums = None
        self.file_indices = self.get_file_indices()
        self._transformed_when_prestacked = False  # an ugly flag to prevent
        # the dataset from getting normalized twice when setting the transform
        # before prestacking
        self.transform = transform
        with h5py.File(self.filepath, 'r') as f:
            assert len(f['features']) == len(f['labels'])
            # calculate number of samples
            self.n_samples = len(f['features'])//decimation
            # calculate number of features
            if self.features is None:
                self._n_current_features = f['features'].shape[1]
                self.feature_indices = [(0, self._n_current_features)]
            else:
                self.feature_indices = self.get_feature_indices()
                self._n_current_features = sum(j-i for i, j in
                                               self.feature_indices)
            if dct_toggle:
                stack = n_dct
            self.n_features = self._n_current_features*(stack + 1)
            # calculate number of labels
            if self.labels is None:
                self.n_labels = f['labels'].shape[1]
                self.label_indices = [(0, self.n_labels)]
            else:
                self.label_indices = self.get_label_indices()
                self.n_labels = sum(j-i for i, j in self.label_indices)
            if self.load:
                logging.info('Loading dataset into memory')
                self.datasets = (f['features'][:], f['labels'][:])
                self.file_nums = f['indexes'][:]
                if self.prestack:
                    logging.info('Prestacking')
                    self.datasets = self[:]
                    self._prestacked = True
                    if self.transform:
                        self._transformed_when_prestacked = True

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if self._transformed_when_prestacked:
            raise ValueError(
                'cannot set transform attribute anymore as the dataset was '
                'already prestacked using the current transform object.'
            )
        self._transform = value

    def get_feature_indices(self):
        pipes_path = os.path.join(self.dirpath, 'pipes.pkl')
        with open(pipes_path, 'rb') as f:
            featureExtractor = pickle.load(f)['featureExtractor']
        names = featureExtractor.features
        indices = featureExtractor.indices
        indices_dict = {name: lims for name, lims in zip(names, indices)}
        if 'itd_ic' in indices_dict.keys():
            start, end = indices_dict.pop('itd_ic')
            step = (end - start)//2
            indices_dict['itd'] = (start, start+step)
            indices_dict['ic'] = (start+step, end)
        feature_indices = [indices_dict[feature] for feature in self.features]
        return feature_indices

    def get_label_indices(self):
        pipes_path = os.path.join(self.dirpath, 'pipes.pkl')
        with open(pipes_path, 'rb') as f:
            labelExtractor = pickle.load(f)['labelExtractor']
        names = labelExtractor.labels
        indices = labelExtractor.indices
        indices_dict = {name: lims for name, lims in zip(names, indices)}
        label_indices = [indices_dict[label] for label in self.labels]
        return label_indices

    def get_file_indices(self):
        metadatas_path = os.path.join(self.dirpath, 'mixture_info.json')
        metadatas = read_json(metadatas_path)
        indices = [item['dataset_indices'] for item in metadatas]
        return indices

    def __getitem__(self, index):
        if self.datasets is None:
            f = h5py.File(self.filepath, 'r')
            if self.load:
                self.datasets = (f['features'][:], f['labels'][:])
                self.file_nums = f['indexes'][:]
                if self.prestack:
                    self.datasets = self[:]
                    self._prestacked = True
                    if self.transform:
                        self._transformed_when_prestacked = True
            else:
                self.datasets = (f['features'], f['labels'])
                self.file_nums = f['indexes']
        if isinstance(index, int):
            file_num = self.file_nums[index]
            if self._prestacked:
                x, y = self.datasets[0][index], self.datasets[1][index]
                if not self._transformed_when_prestacked and self.transform:
                    if isinstance(self.transform, ResursiveTensorStandardizer):
                        if index == 0 or self.file_nums[index-1] != file_num:
                            self.transform.reset()
                    elif isinstance(self.transform, StateTensorStandardizer):
                        self.transform.set_state(file_num)
                    x = self.transform(x)
            else:
                # features
                x = np.empty(self.n_features)
                # decimate
                index *= self.decimation
                # frame at current time index
                count = 0
                for i, j in self.feature_indices:
                    i_ = count
                    j_ = count+j-i
                    x[i_:j_] = self.datasets[0][index, i:j]
                    count = j_
                # frames at previous time indexes
                if self.stack > 0:
                    x_context = np.zeros((self.stack,
                                          self._n_current_features))
                    # first find the starting index of the current file
                    i_file_start, _ = self.file_indices[file_num]
                    # then add context stacking
                    for k in range(self.stack):
                        # if context overlaps previous file then replicate
                        i_lag = max(index-k-1, i_file_start)
                        count_context_k = 0
                        for i, j in self.feature_indices:
                            i_ = count_context_k
                            j_ = count_context_k+j-i
                            x_context[k, i_:j_] = self.datasets[0][i_lag, i:j]
                            count_context_k = j_
                    # perform dct
                    if self.dct_toggle:
                        x_context = dct(x_context, self.n_dct)
                    x[count:] = x_context.flatten()
                if self.transform:
                    if isinstance(self.transform, ResursiveTensorStandardizer):
                        if index == 0 or self.file_nums[index-1] != file_num:
                            self.transform.reset()
                    elif isinstance(self.transform, StateTensorStandardizer):
                        self.transform.set_state(file_num)
                    x = self.transform(x)
                # labels
                y = np.empty(self.n_labels)
                count = 0
                for i, j in self.label_indices:
                    i_ = count
                    j_ = count+j-i
                    y[i_:j_] = self.datasets[1][index, i:j]
                    count = j_
        elif isinstance(index, slice):
            indexes = range(self.n_samples)[index]
            x = np.empty((len(indexes), self.n_features))
            y = np.empty((len(indexes), self.n_labels))
            for i, j in enumerate(indexes):
                x[i, :], y[i, :] = self.__getitem__(j)
        else:
            raise ValueError(f'{type(self).__name__} does not support '
                             f'{type(index).__name__} indexing')
        return x, y

    def __len__(self):
        return self.n_samples


class BreverDataset(torch.utils.data.Dataset):
    """
    Base dataset for all datasets. Reads mixtures from a created dataset.
    Mixtures can be loaded as segments of fixed length or entirely by setting
    segment_length=-1. When a fixed segment is used, the last samples in each
    mixture are dropped.

    Should be subclassed for post-processing by re-implementing the post_proc
    method.
    """
    def __init__(self, path, segment_length=-1, overlap_length=0,
                 components=['foreground', 'background']):
        self.path = path
        self.segment_length = segment_length
        self.overlap_length = overlap_length
        self.components = components
        self.segment_info = self.get_segment_info()
        self.preloaded_data = None

    def build_paths(self, mix_idx):
        mix_dir = os.path.join(self.path, 'audio')
        mix_path = os.path.join(mix_dir, f'{mix_idx:05d}_mixture.wav')
        comp_paths = []
        for component in self.components:
            comp_path = os.path.join(mix_dir, f'{mix_idx:05d}_{component}.wav')
            comp_paths.append(comp_path)
        return mix_path, comp_paths

    def get_segment_info(self):
        mix_lengths = self.get_mix_lengths()
        segment_info = []
        if self.segment_length == -1:
            for mix_idx, mix_length in enumerate(mix_lengths):
                segment_info.append((mix_idx, (0, mix_length)))
        else:
            hop_length = self.segment_length - self.overlap_length
            for mix_idx, mix_length in enumerate(mix_lengths):
                n_segments = (mix_length - self.segment_length)//hop_length + 1
                for segment_idx in range(n_segments):
                    start = segment_idx*hop_length
                    end = start + self.segment_length
                    segment_info.append((mix_idx, (start, end)))
        return segment_info

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
        for idx, length in lengths:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch_list.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch_list.append(batch)
        return batch_list

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
        framer_kwargs={},
        filterbank_kwargs={},
    ):
        super().__init__(path, components=['foreground', 'background'])
        self.stacks = stacks
        self.decimation = decimation
        self.feature_extractor = FeatureExtractor(features)
        self.framer = Framer(**framer_kwargs)
        self.filterbank = Filterbank(**filterbank_kwargs)

    def post_proc(self, data, target, return_filter_output=False):
        data = torch.stack([data, *target])  # (sources, channels, time)
        filt = self.filterbank(data)  # (filts, sources, channels, time)
        data = torch.from_numpy(filt).float()
        data = self.framer(data)  # (filts, sources, channels, frames, samples)
        data = data.numpy()
        mix = data[:, 0, :, :, :]
        foreground = data[:, 1, :, :, :]
        background = data[:, 2, :, :, :]
        # features
        data = self.feature_extractor(mix)  # (features, frames)
        data = torch.from_numpy(data).float()
        data = self.stack(data)
        data = self.decimate(data)
        # labels
        target = self.irm(foreground, background)  # (labels, frames)
        target = torch.from_numpy(target)
        target = self.decimate(target)
        if return_filter_output:
            return data, target, filt
        else:
            return data, target

    def irm(self, target, masker):
        # (filters, channels, frames, samples)
        target = np.mean(target**2, axis=(1, 3))  # (filters, frames)
        masker = np.mean(masker**2, axis=(1, 3))  # (filters, frames)
        return (1 + masker/(target+eps))**-0.5

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
        return [self.framer.count(x) for x in super().item_lengths]

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


class Framer:
    def __init__(self, frame_length=512, hop_length=256):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, x):
        frames = self.count(x.shape[-1])
        padding = (frames - 1)*self.hop_length + self.frame_length - len(x)
        x = F.pad(x, (0, padding))
        return x.unfold(-1, size=self.frame_length, step=self.hop_length)

    def count(self, length):

        def ceil(a, b):
            return -(a//-b)

        return ceil(length - self.frame_length, self.hop_length) + 1


class ConvTasNetDataset(BreverDataset):
    def __init__(self, path, components=['foreground', 'background']):
        super().__init__(path, components=components)

    def post_proc(self, data, target):
        data = data.mean(axis=-2)
        target = target.mean(axis=-2)
        return data, target
