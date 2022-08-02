import random

import numpy as np
import torch


class BreverBatchSampler(torch.utils.data.Sampler):
    """
    Base class for all samplers.
    """
    def __init__(self, dataset, drop_last=False, shuffle=True, seed=0,
                 dynamic=False, sort=False):
        self.__pre_init__(dataset, drop_last, shuffle, seed, dynamic, sort)
        self.generate_batches()

    def __pre_init__(self, dataset, drop_last, shuffle, seed, dynamic, sort):
        self.dataset = dataset
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.dynamic = dynamic
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
        dataset = self.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
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


class _BaseBatchSampler(BreverBatchSampler):
    """
    Base class for the random and sorted batch samplers
    """
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True,
                 seed=0, dynamic=False, sort=False):
        super().__pre_init__(dataset, drop_last, shuffle, seed, dynamic, sort)
        self.batch_size = self.segment_to_item_length(batch_size)
        self.generate_batches()

    def _new_batch(self, batch, item_length):
        output = False
        if self.dynamic:
            if item_length > self.batch_size:
                raise ValueError('found an item that is longer than the '
                                 'dynamic batch size')
            batch_length = max(item[1] for item in batch) if batch else 0
            if (len(batch)+1)*max(item_length, batch_length) > self.batch_size:
                output = True
        elif len(batch)+1 > self.batch_size:
            output = True
        return output

    def _generate_batches(self, indices):
        self.batches = []
        batch = []
        for i in indices:
            item_idx, item_length = self._item_lengths[i]
            if self._new_batch(batch, item_length):
                self.batches.append(batch)
                batch = []
                batch.append((item_idx, item_length))
            else:
                batch.append((item_idx, item_length))
        if len(batch) > 0 and not self.drop_last:
            self.batches.append(batch)


class RandomBatchSampler(_BaseBatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True,
                 seed=0, dynamic=False):
        super().__init__(dataset, batch_size, drop_last=drop_last,
                         shuffle=shuffle, seed=seed, dynamic=dynamic,
                         sort=False)


class SortedBatchSampler(_BaseBatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True,
                 seed=0, dynamic=False):
        super().__init__(dataset, batch_size, drop_last=drop_last,
                         shuffle=shuffle, seed=seed, dynamic=dynamic,
                         sort=True)


class BucketBatchSampler(BreverBatchSampler):
    """
    Items of similar length are grouped into buckets. Batches are formed with
    items from the same bucket. This attempts to minimize both the batch size
    variability and the amount of padding while keeping some randomness.

    Inspired from code by Speechbrain under Apache-2.0 License:
    https://github.com/speechbrain/speechbrain/blob/b5d2836e3d0eabb541c5bdbca16fb00c49cb62a3/speechbrain/dataio/sampler.py#L305
    """
    def __init__(self, dataset, batch_size, num_buckets=10, drop_last=False,
                 shuffle=True, seed=0, dynamic=False):
        super().__pre_init__(dataset, drop_last, shuffle, seed, dynamic,
                             sort=False)
        max_length = max(item[1] for item in self._item_lengths)
        batch_size = self.segment_to_item_length(batch_size)
        max_length = self.segment_to_item_length(max_length)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_buckets = num_buckets

        self.right_bucket_limits = np.linspace(
            max_length/num_buckets, max_length, num_buckets,
        )
        if self.dynamic:
            self.bucket_batch_lengths = batch_size//self.right_bucket_limits
        else:
            self.bucket_batch_lengths = [batch_size]*self.num_buckets

        self.generate_batches()

    def _generate_batches(self, indices):
        batches = [[] for _ in range(self.num_buckets)]
        current_batches = [[] for _ in range(self.num_buckets)]
        for i in indices:
            item_idx, item_length = self._item_lengths[i]
            bucket_idx = np.searchsorted(
                self.right_bucket_limits, item_length,
            )
            if bucket_idx == self.num_buckets:
                if item_length == self.max_length:
                    bucket_idx -= 1
                else:
                    raise ValueError('found an item that is longer than the '
                                     'maximum item length')
            current_batches[bucket_idx].append((item_idx, item_length))
            if len(current_batches[bucket_idx]) \
                    == self.bucket_batch_lengths[bucket_idx]:
                batches[bucket_idx].append(current_batches[bucket_idx])
                current_batches[bucket_idx] = []
            elif len(current_batches[bucket_idx]) \
                    > self.bucket_batch_lengths[bucket_idx]:
                raise ValueError('maximum number of items in bucket exceeded')
        if not self.drop_last:
            for bucket_idx, batch in enumerate(current_batches):
                if len(batch) > 0:
                    batches[bucket_idx].append(batch)
        self.batches = [item for batch in batches for item in batch]


def get_batch_sampler(name, batch_size, fs, num_buckets, dynamic):
    if dynamic:
        batch_size = round(batch_size*fs)
    else:
        if not(isinstance(batch_size, int)
                or (isinstance(batch_size, float)
                    and batch_size == int(batch_size))):
            raise ValueError(f"batch_size must be int, got {batch_size}")
    if name == 'bucket':
        batch_sampler_class = BucketBatchSampler
        kwargs = {
            'batch_size': batch_size,
            'dynamic': dynamic,
            'num_buckets': num_buckets
        }
    elif name == 'random':
        batch_sampler_class = RandomBatchSampler
        kwargs = {
            'batch_size': batch_size,
            'dynamic': dynamic,
        }
    elif name == 'sorted':
        batch_sampler_class = SortedBatchSampler
        kwargs = {
            'batch_size': batch_size,
            'dynamic': dynamic,
        }
    else:
        raise ValueError(f'Unrecognized batch sampler, got {name}')
    return batch_sampler_class, kwargs
