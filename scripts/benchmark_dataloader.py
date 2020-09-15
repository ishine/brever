import time
import argparse
import os

import torch

from brever.config import defaults
from brever.pytorchtools import H5Dataset
from brever.modelmanagement import get_feature_indices, get_file_indices


def main(args):
    dataset_path = os.path.join(args.input, 'dataset.hdf5')
    feature_indices = get_feature_indices(args.input, args.features)
    file_indices = get_file_indices(args.input)

    dataset = H5Dataset(
        filepath=dataset_path,
        load=args.load,
        stack=args.stacks,
        decimation=args.decimation,
        feature_indices=feature_indices,
        file_indices=file_indices,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        drop_last=True,
    )

    start_time = time.time()
    for data, target in dataloader:
        pass
    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {elapsed_time:.2f}')


if __name__ == '__main__':
    config = defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--load', dest='load', action='store_true')
    parser.add_argument('--no-load', dest='load', action='store_false')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--stacks', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--decimation', type=int)
    parser.add_argument('--features', type=lambda x: set(x.split(' ')))
    parser.set_defaults(
        load=config.POST.LOAD,
        shuffle=config.MODEL.SHUFFLE,
        workers=config.MODEL.NWORKERS,
        stacks=config.POST.STACK,
        batch_size=config.MODEL.BATCHSIZE,
        decimation=config.POST.DECIMATION,
        features=config.POST.FEATURES,
    )
    args = parser.parse_args()
    main(args)
