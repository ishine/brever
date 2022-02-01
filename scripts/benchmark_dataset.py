import argparse
import time

from brever.data import BreverBatchSampler, BreverDataLoader, DNNDataset


def main():
    print('Initializing dataset')
    dataset = DNNDataset(
        path=args.input,
        features=args.features,
    )

    print('Initializing batch sampler')
    batch_sampler = BreverBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
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
    parser.add_argument('--features', nargs='+')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int)
    parser.set_defaults(
        features=['fbe'],
        batch_size=1,
        epochs=1,
        workers=0,
    )
    args = parser.parse_args()
    main()
