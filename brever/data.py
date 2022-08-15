import json
import os
import tarfile
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf


eps = np.finfo(float).eps


class TarArchiveInterface:
    """
    Hats off to João F. Henriques for his tarfile PyTorch dataset:
    https://github.com/jotaf98/simple-tar-dataset

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
    """
    def __init__(self, path, segment_length=4.0, overlap_length=0.0, fs=16e3,
                 components=['foreground', 'background'],
                 segment_strategy='pass', tar=True, model=None,
                 dynamic_batch_size=None):
        self.path = path
        self.segment_length = round(segment_length*fs)
        self.overlap_length = round(overlap_length*fs)
        self.components = components
        self.segment_strategy = segment_strategy
        if dynamic_batch_size is not None:
            dynamic_batch_size = round(dynamic_batch_size*fs)
        self.dynamic_batch_size = dynamic_batch_size
        if tar:
            self.archive = TarArchiveInterface(os.path.join(path, 'audio.tar'))
        else:
            self.archive = None
        self.segment_info = self.get_segment_info()
        self.preloaded_data = None
        self._item_lengths = None
        self.model = model

    def get_file(self, name):
        if self.archive is None:
            file = open(os.path.join(self.path, name), 'rb')
        else:
            file = self.archive.get_file(name.replace('\\', '/'))
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
        self._duration = sum(mix_lengths)

        if self.segment_length == 0.0 and self.dynamic_batch_size is not None:
            max_mix_length = max(mix_lengths)
            if self.dynamic_batch_size < max_mix_length:
                logging.warning('The dynamic batch size is smaller than '
                                'the maximum mixture length. Setting the '
                                'segment length to the dynamic batch size '
                                f'({self.dynamic_batch_size}).')
                self.segment_length = self.dynamic_batch_size

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
        if self.model is None:
            item_length = segment_length
        else:
            item_length = self.model.segment_to_item_length(segment_length)
        return item_length

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
            if self.model is not None:
                data, target = self.model.pre_proc(data, target)
        return data, target

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

    def get_statistics(self):
        mean, pow_mean = 0, 0
        for i in range(len(self)):
            data, _ = self[i]
            dim = tuple(range(1, data.ndim))
            mean += data.mean(dim, keepdim=True)
            pow_mean += data.pow(2).mean(dim, keepdim=True)
        mean /= len(self)
        pow_mean /= len(self)
        var = pow_mean - mean.pow(2)
        std = var.sqrt()
        return mean, std


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
