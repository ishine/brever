import time
import argparse

import torch

from brever.config import defaults
from brever.pytorchtools import H5Dataset


def main(args):
    dataset = H5Dataset(
        dirpath=args.input,
        load=args.load,
        stack=args.stacks,
        decimation=args.decimation,
        prestack=args.prestack,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        drop_last=True,
    )

    start_time = time.time()
    for i in range(args.epochs):
        start_time_i = time.time()
        for data, target in dataloader:
            pass
        elapsed_time_i = time.time() - start_time_i
        print(f'Elapsed time on epoch {i}: {elapsed_time_i:.2f}')
    elapsed_time = time.time() - start_time
    print(f'Total elapsed time: {elapsed_time:.2f}')


if __name__ == '__main__':
    config = defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--load', dest='load', action='store_true')
    parser.add_argument('--no-load', dest='load', action='store_false')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    parser.add_argument('--prestack', dest='prestack', action='store_true')
    parser.add_argument('--no-prestack', dest='prestack', action='store_false')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--stacks', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--decimation', type=int)
    parser.add_argument('--features', type=lambda x: set(x.split(' ')))
    parser.add_argument('--epochs', type=int)
    parser.set_defaults(
        load=config.POST.LOAD,
        shuffle=config.MODEL.SHUFFLE,
        prestack=config.POST.PRESTACK,
        workers=config.MODEL.NWORKERS,
        stacks=config.POST.STACK,
        batch_size=config.MODEL.BATCHSIZE,
        decimation=config.POST.DECIMATION,
        features=config.POST.FEATURES,
        epochs=10,
    )
    args = parser.parse_args()
    main(args)
