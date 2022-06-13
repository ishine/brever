import numpy as np
import matplotlib.pyplot as plt

from brever.data import BreverDataset
import brever.data
from brever.config import DatasetInitializer


def plot(batches, title=None):
    # batches = sorted(batches, key=lambda x: max(b[1] for b in x))
    fig, ax = plt.subplots()
    i = 0
    for batch in batches:
        x = np.arange(len(batch)) + i
        lengths = [x[1] for x in batch]
        max_length = max(lengths)
        ax.bar(x, [max_length]*len(batch), width=1, color='k')
        ax.bar(x, lengths, width=1)
        i += len(batch)
    if title is not None:
        ax.set_title(title)


def main():
    dset_init = DatasetInitializer()
    path = dset_init.get_path_from_kwargs(
        kind='train',
        speakers={'timit_.*'},
        noises={'dcase_.*'},
        rooms={'surrey_.*'},
        speech_files=[0.0, 0.8],
        noise_files=[0.0, 0.8],
        room_files='even',
        duration=3600,
        seed=0,
    )

    dataset = BreverDataset(
        path,
        segment_length=4.0,
    )
    duration = dataset._duration

    fig, ax = plt.subplots(2, 1)

    for batch_sampler_name, kwargs in {
        'SimpleBatchSampler': {
            'items_per_batch': 4,
        },
        'DynamicSimpleBatchSampler': {
            'max_batch_size': 4*4*16e3,
        },
        'SortedBatchSampler': {
            'items_per_batch': 4,
        },
        'DynamicSortedBatchSampler': {
            'max_batch_size': 4*4*16e3,
        },
        'BucketBatchSampler': {
            'max_batch_size': 4*4*16e3,
            'max_item_length': 4*16e3,
            'num_buckets': 10,
        },
        'StaticBucketBatchSampler': {
            'batch_size': 4,
            'max_item_length': 4*16e3,
            'num_buckets': 10,
        },
    }.items():

        print(batch_sampler_name)

        batch_sampler = getattr(brever.data, batch_sampler_name)
        batch_sampler = batch_sampler(
            dataset=dataset, drop_last=True, **kwargs
        )

        batch_sizes, pad_amounts = batch_sampler.calc_batch_stats()
        print(f'batches: {len(batch_sizes)}')
        print(f'min batch size: {min(batch_sizes)/16e3:.2f}')
        print(f'max batch size: {max(batch_sizes)/16e3:.2f}')
        print(f'batch size std: {np.std(batch_sizes)/16e3:.2f}')
        print(f'total pad amount: {sum(pad_amounts)/duration*100:.2f}%')
        print(f'pad std: {np.std(pad_amounts)/16e3:.2f}')

        plot(batch_sampler.batches, title=batch_sampler_name)

        ax[0].hist(batch_sizes, label=batch_sampler_name, alpha=.5,
                   density=True, bins=np.linspace(0, 260000, 100))
        ax[1].hist(pad_amounts, label=batch_sampler_name, alpha=.5,
                   density=True, bins=np.linspace(0, 120000, 100))

        print('')

    ax[0].set_title('batch size distribution')
    ax[1].set_title('padding distribution')
    ax[0].legend()
    ax[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
