import os
import shutil

from brever.args import DatasetArgParser
from brever.config import DatasetFinder


def main():
    if args.created and args.uncreated:
        raise ValueError('cannot use both --created and --uncreated')

    finder = DatasetFinder()
    matching_dsets, _ = finder.find(kind=args.kind)

    dsets = []
    for dset in matching_dsets:
        mix_info_file = os.path.join(dset, 'mixture_info.json')

        if args.uncreated and os.path.exists(mix_info_file):
            continue
        if args.created and not os.path.exists(mix_info_file):
            continue

        dsets.append(dset)

    if args.pipe:
        print(' '.join(dsets), end='')
    else:
        for dset in dsets:
            print(dset)

    if dsets and args.delete:
        print(f'{len(dsets)} datasets will be deleted.')
        resp = input('Do you want to continue? y/n')
        if resp == 'y':
            for dset in dsets:
                shutil.rmtree(dset)
                print(f'Deleted {dset}')
        else:
            print('No dataset was deleted')


if __name__ == '__main__':
    parser = DatasetArgParser(description='find datasets')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='delete found datasets')
    parser.add_argument('--created', action='store_true',
                        help='only show created datasets')
    parser.add_argument('--uncreated', action='store_true',
                        help='only show created datasets')
    parser.add_argument('--pipe', action='store_true',
                        help='output as one line to pipe to another command')
    parser.add_argument('--kind', choices=['train', 'test'],
                        help='scan train or test subdir')
    args = parser.parse_args()
    main()
