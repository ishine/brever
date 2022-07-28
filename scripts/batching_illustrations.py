import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (3.5, 1)
plt.rcParams['font.size'] = 5
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['patch.linewidth'] = .1
plt.rcParams['axes.linewidth'] = .4
plt.rcParams['grid.linewidth'] = .4
plt.rcParams['xtick.major.size'] = 1
plt.rcParams['xtick.major.width'] = .5
plt.rcParams['ytick.major.size'] = 1
plt.rcParams['ytick.major.width'] = .5


def crop(lengths):
    output = []
    for length in lengths:
        while length > 4.0:
            length -= 4.0
            output.append(4.0)
        output.append(length)
    return output


def plot(lengths, batch_func, filename=None):
    x = np.arange(len(lengths))
    batches = batch_func(lengths)
    plt.figure()
    i = 0
    for i_batch, batch in enumerate(batches):
        x = np.arange(len(batch)) + i
        max_len = max(batch)
        plt.bar(x, batch, width=1, align='edge', edgecolor='k')
        plt.bar(x, [max_len]*len(batch), width=1, color='k', zorder=0,
                align='edge', edgecolor='k')
        i += len(batch)
    plt.xticks([])
    plt.xlim(-2, len(lengths)+2)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)


def batch_simple(lengths, batch_size=4):
    batches = []
    batch = []
    for length in lengths:
        batch.append(length)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    return batches


def batch_dynamic(lengths, batch_size=16):
    batches = []
    batch = []
    batch_width = 0
    for length in lengths:
        if (len(batch)+1)*max(length, batch_width) <= batch_size:
            batch.append(length)
            batch_width = max(length, batch_width)
        else:
            batches.append(batch)
            batch = []
            batch.append(length)
            batch_width = length
    if len(batch) > 0:
        batches.append(batch)
    return batches


def batch_bucket(lengths, batch_size=16, num_buckets=10):
    max_length = max(lengths)
    right_bucket_limits = np.linspace(
        max_length/num_buckets, max_length, num_buckets,
    )
    bucket_batch_lengths = batch_size//right_bucket_limits

    batches = []
    bucket_batches = [[] for _ in range(num_buckets)]
    for length in lengths:
        i_bucket = np.searchsorted(
            right_bucket_limits, length,
        )
        if i_bucket == num_buckets:
            if length == max_length:
                i_bucket -= 1
            else:
                raise ValueError('found an item that is longer than the '
                                 'maximum item length')
        bucket_batches[i_bucket].append(length)
        if len(bucket_batches[i_bucket]) == bucket_batch_lengths[i_bucket]:
            batches.append(bucket_batches[i_bucket])
            bucket_batches[i_bucket] = []
        elif len(bucket_batches[i_bucket]) > bucket_batch_lengths[i_bucket]:
            raise ValueError('bucket maximum number of items exceeded')
    for batch in bucket_batches:
        if len(batch) > 0:
            batches.append(batch)
    return batches


np.random.seed(0)

n_mixtures = 100
mean_length = 4
length_std = 1.5
lengths = mean_length + length_std*np.random.randn(n_mixtures)
# lengths = crop(lengths)

plot(lengths, batch_simple, 'batching_simple.svg')
plot(sorted(lengths), batch_simple, 'batching_simple_sorted.svg')
plot(lengths, batch_dynamic, 'batching_dynamic.svg')
plot(sorted(lengths), batch_dynamic, 'batching_dynamic_sorted.svg')
plot(lengths, batch_bucket, 'batching_bucket.svg')

plt.show()
