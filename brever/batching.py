import random

import numpy as np
import torch


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
    items from the same bucket. This attempts to minimize both the batch size
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


class StaticBucketBatchSampler(BreverBatchSampler):
    """
    Same as BucketBatchSampler but with a batch size defined as the number of
    mixtures instead of the duration
    """
    def __init__(self, dataset, batch_size, max_item_length,
                 num_buckets=10, drop_last=False, shuffle=True, seed=0):
        super().__pre_init__(dataset, drop_last, shuffle, seed, sort=False)
        max_item_length = self.segment_to_item_length(max_item_length)
        self.batch_size = batch_size
        self.max_item_length = max_item_length
        self.num_buckets = num_buckets

        self.right_bucket_limits = np.linspace(
            max_item_length/num_buckets, max_item_length, num_buckets,
        )

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
            if len(bucket_batches[bucket_idx]) == self.batch_size:
                self.batches.append(bucket_batches[bucket_idx])
                bucket_batches[bucket_idx] = []
            elif len(bucket_batches[bucket_idx]) > self.batch_size:
                raise ValueError('bucket maximum number of items exceeded')
        if not self.drop_last:
            for batch in bucket_batches:
                if len(batch) > 0:
                    self.batches.append(batch)


def get_batch_sampler(name, batch_size, fs, num_buckets, segment_length,
                      sorted_):
    if name == 'bucket':
        batch_sampler_class = BucketBatchSampler
        kwargs = {
            'max_batch_size': round(batch_size*fs),
            'max_item_length': round(segment_length*fs),
            'num_buckets': num_buckets
        }
    elif name == 'dynamic':
        if sorted_:
            batch_sampler_class = DynamicSortedBatchSampler
        else:
            batch_sampler_class = DynamicSimpleBatchSampler
        kwargs = {
            'max_batch_size': round(batch_size*fs),
        }
    elif name == 'simple':
        if sorted_:
            batch_sampler_class = SortedBatchSampler
        else:
            batch_sampler_class = SimpleBatchSampler
        if isinstance(batch_size, float) and batch_size != int(batch_size):
            raise ValueError("when using 'simple' batch sampler, batch_size "
                             "must be int or float equal to int, got "
                             f"{batch_size}")
        kwargs = {
            'items_per_batch': batch_size,
        }
    else:
        raise ValueError(f'Unrecognized batch sampler, got {name}')
    return batch_sampler_class, kwargs
