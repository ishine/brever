import argparse
import time

import brever.data


def main():
    print('Initializing dataset')
    dataset = {
        'dnn': brever.data.DNNDataset,
        'convtasnet': brever.data.ConvTasNetDataset,
    }[args.arch]
    kwargs = {
        'dnn': {
            'features': args.features
        },
        'convtasnet': {},
    }[args.arch]
    dataset = dataset(
        args.input,
        segment_length=args.segment_length,
        **kwargs
    )

    print('Initializing batch sampler')
    sampler = {
        'bucket': brever.data.BucketBatchSampler,
    }[args.sampler]
    kwargs = {
        'bucket': {
            'max_batch_size': args.max_batch_size,
            'max_item_length': args.segment_length,
        },
    }[args.sampler]
    sampler = sampler(dataset, **kwargs)

    print('Initializing data loader')
    dataloader = brever.data.BreverDataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
    )

    print('Starting benchmark')
    elapsed_time = 0
    start_time = time.time()
    for i in range(args.epochs):
        for data, target in dataloader:
            pass
        dt = time.time() - start_time - elapsed_time
        elapsed_time = time.time() - start_time
        print(f'Time on epoch {i}: {dt:.2f}')
    print(f'Total time: {elapsed_time:.2f}')
    print(f'Averate time per epoch: {elapsed_time/args.epochs:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark a dataset')
    parser.add_argument('input', help='dataset directory')
    parser.add_argument('--arch')
    parser.add_argument('--features', nargs='+')
    parser.add_argument('--sampler')
    parser.add_argument('--segment-length', type=float)
    parser.add_argument('--max-batch-size', type=float)
    parser.add_argument('--items-per-batch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--fs', type=int)
    parser.set_defaults(
        arch='convtasnet',
        features={'logfbe'},
        sampler='bucket',
        segment_length=4.0,
        max_batch_size=16.0,
        items_per_batch=4,
        epochs=1,
        workers=0,
        fs=16e3,
    )
    args = parser.parse_args()
    args.segment_length = int(round(args.segment_length)*args.fs)
    args.max_batch_size = int(round(args.max_batch_size)*args.fs)
    main()
