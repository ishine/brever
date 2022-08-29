import os
import shutil

from brever.args import ModelArgParser
from brever.config import ModelFinder, DatasetFinder


def main():
    all_dsets, _ = DatasetFinder().find(kind='train')
    finder = ModelFinder()

    dsets = []
    for train_path in all_dsets:
        models, _ = finder.find(train_path=train_path)
        delete = True
        for model in models:
            loss_file = os.path.join(model, 'losses.npz')
            if not os.path.exists(loss_file):
                delete = False
        if delete:
            dsets.append(train_path)

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
    parser = ModelArgParser(req=False, description='find deletable datasets')
    parser.add_argument('--delete', action='store_true',
                        help='delete found datasets')
    parser.add_argument('--pipe', action='store_true',
                        help='output as one line to pipe to another command')
    args = parser.parse_args()
    main()
