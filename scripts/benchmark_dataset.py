import argparse
import time

from brever.data import BreverDataset, BreverDataLoader
from brever.batching import get_batch_sampler


def main():
    print('Initializing dataset')
    kwargs = {}
    if args.sources is not None:
        kwargs['components'] = args.sources
    if args.dynamic:
        kwargs['dynamic_batch_size'] = args.batch_size
    dataset = BreverDataset(
        path=args.input,
        segment_length=args.segment_length,
        fs=args.fs,
        **kwargs,
    )

    print('Initializing batch sampler')
    batch_sampler_class, kwargs = get_batch_sampler(
        name=args.sampler,
        batch_size=args.batch_size,
        fs=args.fs,
        num_buckets=args.buckets,
        dynamic=args.dynamic,
    )
    batch_sampler = batch_sampler_class(
        dataset=dataset,
        **kwargs,
    )

    print('Initializing data loader')
    dataloader = BreverDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
    )

    print('Starting benchmark')
    elapsed_time = 0
    start_time = time.time()
    for i in range(args.epochs):
        for data, target, lengths in dataloader:
            pass
        dt = time.time() - start_time - elapsed_time
        elapsed_time = time.time() - start_time
        print(f'Time on epoch {i}: {dt:.2f}')
    print(f'Total time: {elapsed_time:.2f}')
    print(f'Averate time per epoch: {elapsed_time/args.epochs:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark a dataset')
    parser.add_argument('input', help='dataset directory')
    parser.add_argument('--segment-length', type=float, default=0.0)
    parser.add_argument('--sampler', type=str, default='bucket')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--batch-size', type=float, default=1)
    parser.add_argument('--buckets', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--fs', type=int, default=16e3)
    parser.add_argument('--sources', type=str, nargs='+')
    args = parser.parse_args()
    main()
