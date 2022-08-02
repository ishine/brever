import numpy as np
import matplotlib.pyplot as plt


from brever.batching import get_batch_sampler
from brever.config import DatasetInitializer
from brever.data import BreverDataset


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


class Dataset:
    def __init__(self, item_lengths):
        self.item_lengths = item_lengths

    def segment_to_item_length(self, length):
        return length


def plot(sampler_name, dynamic, batch_size, filename=None):
    sampler, kwargs = get_batch_sampler(sampler_name, batch_size, fs,
                                        num_buckets, dynamic)
    sampler = sampler(dataset, **kwargs)
    sampler.generate_batches()

    x = np.arange(len(item_lengths))
    plt.figure()
    i = 0
    for i_batch, batch_ in enumerate(sampler.batches):
        batch = [item[1] for item in batch_]
        x = np.arange(len(batch)) + i
        max_len = max(batch)
        plt.bar(x, batch, width=1, align='edge', edgecolor='k')
        plt.bar(x, [max_len]*len(batch), width=1, color='k', zorder=0,
                align='edge', edgecolor='k')
        i += len(batch)
    plt.xticks([])
    plt.xlim(-2, len(item_lengths)+2)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)


np.random.seed(0)

fs = 1
n_mixtures = 100
max_length = 8.0*fs
num_buckets = 3

item_lengths = np.random.uniform(0.0, max_length, n_mixtures)
dataset = Dataset(item_lengths)

plot(
    sampler_name='random',
    dynamic=False,
    batch_size=4,
    filename='batching/batching_random_fixed.svg',
),
plot(
    sampler_name='random',
    dynamic=True,
    batch_size=16.0,
    filename='batching/batching_random_dynamic.svg',
),
plot(
    sampler_name='sorted',
    dynamic=False,
    batch_size=4,
    filename='batching/batching_sorted_fixed.svg',
),
plot(
    sampler_name='sorted',
    dynamic=True,
    batch_size=16.0,
    filename='batching/batching_sorted_dynamic.svg',
),
plot(
    sampler_name='bucket',
    dynamic=False,
    batch_size=4,
    filename='batching/batching_bucket_fixed.svg',
),
plot(
    sampler_name='bucket',
    dynamic=True,
    batch_size=16.0,
    filename='batching/batching_bucket_dynamic.svg',
),


dset_init = DatasetInitializer(batch_mode=True)

p_train = dset_init.get_path_from_kwargs(
    kind='train',
    speakers={'libri_.*'},
    noises={'dcase_.*'},
    rooms={'surrey_.*'},
    speech_files=[0.0, 0.8],
    noise_files=[0.0, 0.8],
    room_files='even',
    duration=36000,
    seed=0,
)

dataset = BreverDataset(
    path=p_train,
    segment_length=4.0,
    fs=16000,
)

plt.figure()
plt.bar(
    np.arange(len(dataset.item_lengths)),
    sorted(dataset.item_lengths),
    width=1,
)


plt.show()
