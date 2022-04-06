import itertools
import time

import matplotlib.pyplot as plt

from brever.data import BreverDataset, BreverDataLoader, BreverBatchSampler
from brever.config import DatasetInitializer


fs = 16e3


dset_init = DatasetInitializer()
path = dset_init.get_path_from_kwargs(
    kind='train',
    speakers={'libri_.*'},
    noises={'dcase_.*'},
    rooms={'surrey_.*'},
    speech_files=[0.0, 0.8],
    noise_files=[0.0, 0.8],
    room_files='even',
    duration=3600,
    seed=0,
)

fs = 16e3

# calculate total length and plot distribution
lenghts = []
dset = BreverDataset(path, segment_length=0)
for x, y in dset:
    lenghts.append(x.size(-1))
total_length = sum(lenghts)
print(f'true dataset total length: {total_length/fs}')
plt.figure()
plt.hist(lenghts, bins=20)


def format_(duration):
    duration = round(duration/fs)
    return f'{duration} ({round((duration/total_length)*100)}%)'


hyperparams = {
    'batch_size': [1, 8, 16],
    'segment_length': [0.0, 1.0, 4.0],
    'segment_strategy': ['drop', 'pass'],
}
results = []
for values in itertools.product(*hyperparams.values()):
    kwargs = dict(zip(hyperparams.keys(), values))
    print(kwargs)

    dataset = BreverDataset(
        path,
        segment_length=kwargs['segment_length'],
        segment_strategy=kwargs['segment_strategy']
    )
    batch_sampler = BreverBatchSampler(
        dataset=dataset,
        batch_size=kwargs['batch_size'],
    )
    dataloader = BreverDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
    )

    dataloader_length = 0
    start_time = time.time()
    for x, y in dataloader:
        batch_size, _, length = x.shape
        dataloader_length += batch_size*length
    duration = time.time() - start_time
    print(f'dataset padding: {format_(dataset._pad_amount)}')
    print(f'dataset dropping: {format_(dataset._drop_amount)}')
    print(f'batching padding: {format_(batch_sampler._pad_amount)}')
    print(f'effective total length: {format_(dataloader_length)}')
    print(f'epoch duration: {round(duration)}')
    assert total_length \
        + dataset._pad_amount \
        - dataset._drop_amount \
        + batch_sampler._pad_amount == dataloader_length

    results.append({
        'params': kwargs,
        'results': {
            'segment_pad': round(dataset._pad_amount/fs),
            'segment_drop': round(dataset._pad_amount/fs),
            'collate_pad': round(batch_sampler._pad_amount/fs),
            'epoch time': round(duration),
        },
    })


import json
with open('batching.json', 'w') as f:
    json.dump(results, f)


plt.show()
