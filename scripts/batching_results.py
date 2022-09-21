import itertools
import os
import h5py
import logging
import argparse

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer, get_config
from brever.display import pretty_table
from brever.data import BreverDataset
from brever.batching import get_batch_sampler
from brever.models import initialize_model


plt.rcParams['font.size'] = 5
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['patch.linewidth'] = .1
plt.rcParams['axes.linewidth'] = .4
plt.rcParams['grid.linewidth'] = .4
plt.rcParams['lines.linewidth'] = .6
plt.rcParams['xtick.major.size'] = 1
plt.rcParams['xtick.major.width'] = .5
plt.rcParams['ytick.major.size'] = 1
plt.rcParams['ytick.major.width'] = .5


DATABASES = [
    {
        'kwarg': 'speakers',
        'databases': [
            'timit_.*',
            'libri_.*',
            'wsj0_.*',
            'clarity_.*',
            'vctk_.*',
        ],
    },
    {
        'kwarg': 'noises',
        'databases': [
            'dcase_.*',
            'noisex_.*',
            'icra_.*',
            'demand',
            'arte',
        ],
    },
    {
        'kwarg': 'rooms',
        'databases': [
            'surrey_.*',
            'ash_.*',
            'bras_.*',
            'catt_.*',
            'avil_.*',
        ],
    },
]
METRICS = [
    {
        'name': 'PESQ',
        'scale': 10,
    },
    {
        'name': 'STOI',
        'scale': 100,
    },
    {
        'name': 'SNR',
        'scale': 1,
    },
]
N_METRICS = len(METRICS)


def fmt_time(time_):
    h, m = int(time_//3600), int((time_ % 3600)//60)
    return fr'{h} h {m:02d} m'


def fmt_score(score, is_max):
    score = f"{score:.2f}"
    if is_max:
        score = fr"\textbf{{{score}}}"
    return score


def fmt_memory(memory):
    memory = round(memory/1e9, 1)
    return f'{memory} GB'


def fmt_padding(padding_fraction):
    return f'{round(padding_fraction*100, 1)}\\%'


def get_padding(model):
    # load model config
    config_path = os.path.join(model, 'config.yaml')
    config = get_config(config_path)

    # initialize model
    model = initialize_model(config)

    # initialize dataset
    kwargs = {}
    if hasattr(config.MODEL, 'SOURCES'):
        kwargs['components'] = config.MODEL.SOURCES
    if config.TRAINING.BATCH_SAMPLER.DYNAMIC:
        kwargs['dynamic_batch_size'] = config.TRAINING.BATCH_SAMPLER.BATCH_SIZE
    dataset = BreverDataset(
        path=config.TRAINING.PATH,
        segment_length=config.TRAINING.SEGMENT_LENGTH,
        fs=config.FS,
        model=model,
        silent=True,
        **kwargs,
    )

    # train val split
    val_length = int(len(dataset)*config.TRAINING.VAL_SIZE)
    train_length = len(dataset) - val_length
    train_split, val_split = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )

    # initialize samplers
    batch_sampler_class, kwargs = get_batch_sampler(
        name=config.TRAINING.BATCH_SAMPLER.WHICH,
        batch_size=config.TRAINING.BATCH_SAMPLER.BATCH_SIZE,
        fs=config.FS,
        num_buckets=config.TRAINING.BATCH_SAMPLER.NUM_BUCKETS,
        dynamic=config.TRAINING.BATCH_SAMPLER.DYNAMIC,
    )
    train_batch_sampler = batch_sampler_class(
        dataset=train_split,
        **kwargs,
    )

    batch_sizes, pad_amounts = train_batch_sampler.calc_batch_stats()
    fraction = sum(pad_amounts)/(sum(batch_sizes)-sum(pad_amounts))
    return fraction


def complement(idx_list):
    return [i for i in range(5) if i not in idx_list]


def build_test_index(index, dims):
    test_index = [complement(index[dim]) for dim in range(3)]
    for dim in dims:
        test_index[dim] = index[dim]
    return test_index


class TrainCurvePlotter:
    def __init__(self):
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.counter = 0

    def plot(self, ax, model, batch_size, batch_sampler, dynamic):
        path = os.path.join(model, 'losses.npz')
        data = np.load(path)
        batch_type = 'dynamic' if dynamic else 'fixed'
        label = (
            f'batch_sampler={batch_sampler}, '
            f'batch_size={batch_size}, '
            f'batch_type={batch_type}'
        )
        color = self.colors[self.counter]
        l, = ax.plot(data['train'], label=label, color=color)
        _, = ax.plot(data['val'], '--', color=color)

    def next_color(self):
        self.counter = (self.counter + 1) % len(self.colors)


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')

    fig, ax = plt.subplots(figsize=(4, 3))

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

    # plot distribution
    dataset = BreverDataset(
        path=p_train,
        segment_length=0.0,
        fs=16000,
    )
    lengths = np.array(dataset.item_lengths)/16e3
    plt.figure(figsize=(3.5, 1))
    plt.hist(lengths, bins=25)
    plt.xlabel('Mixture length (s)')
    plt.ylabel('Count')
    plt.grid()
    # plt.savefig('dist.svg', bbox_inches='tight', pad_inches=0)

    p_test = dset_init.get_path_from_kwargs(
        kind='test',
        speakers={'libri_.*'},
        noises={'dcase_.*'},
        rooms={'surrey_.*'},
        speech_files=[0.8, 1.0],
        noise_files=[0.8, 1.0],
        room_files='odd',
        duration=3600,
        seed=42,
    )

    train_curver_plotter = TrainCurvePlotter()

    def plot_models(scores, batch_sizes, batch_type):
        stats, scores = scores[:, :3], scores[:, 3:]
        max_score = np.nanmax(scores, axis=0)
        for stat, score, batch_size in zip(stats, scores, batch_sizes):
            plot_model(stat, score, batch_size, max_score, batch_type)

    def plot_model(stat, score, batch_size, max_score, batch_type):
        is_max = score == max_score
        if batch_type == 'seq.':
            batch_size = int(batch_size)
        batch_size = f'{batch_size} {batch_type}'
        print(fr' & {batch_size}', end='')
        print(fr' & {fmt_time(stat[0])}', end='')
        print(fr' & {fmt_memory(stat[1])}', end='')
        print(fr' & {fmt_padding(stat[2])}', end='')
        for i_offset in [0, 3]:  # allows to grab matched or mismatched score
            for i_m, m in enumerate(METRICS):
                i_s = i_m + i_offset
                score_fmt = fmt_score(score[i_s]*m["scale"], is_max[i_s])
                print(fr' & {score_fmt}', end='')
        print(r'\\')

    def get_test_dsets(index):
        test_paths = []
        for i, j, k in itertools.product(*index):
            test_path = dset_init.get_path_from_kwargs(
                kind='test',
                speakers={DATABASES[0]['databases'][i]},
                noises={DATABASES[1]['databases'][j]},
                rooms={DATABASES[2]['databases'][k]},
                speech_files=[0.8, 1.0],
                noise_files=[0.8, 1.0],
                room_files='odd',
                duration=3600,
                seed=42,
            )
            test_paths.append(test_path)
        return test_paths

    train_index = [[1], [0], [0]]
    test_idx = build_test_index(train_index, [])
    p_test_mismatch = get_test_dsets(test_idx)

    def load_score(model):
        if not os.path.exists(model):
            raise FileNotFoundError(f'model {model} does not exist')
        filename = os.path.join(model, 'scores.hdf5')
        h5f = h5py.File(filename)
        if p_test not in h5f.keys():
            raise KeyError(f'model {model} is not tested on {p_test}')
        metric_idx = [
            list(h5f['metrics'].asstr()).index(m['name']) for m in METRICS
        ]
        matched_scores = h5f[p_test][:, metric_idx, :].mean(axis=0)
        matched_scores = matched_scores[:, 1] - matched_scores[:, 0]

        mismatched_scores = []
        for p_test_mis in p_test_mismatch:
            if p_test_mis not in h5f.keys():
                raise KeyError(f'model {model} is not tested on {p_test_mis}')
            _scores = h5f[p_test_mis][:, metric_idx, :].mean(axis=0)
            _scores = _scores[:, 1] - _scores[:, 0]
            mismatched_scores.append(_scores)
        mismatched_scores = np.mean(mismatched_scores, axis=0)

        state = torch.load(os.path.join(model, 'checkpoint.pt'),
                           map_location='cpu')
        stats = np.array([
            state["time_spent"],
            state["max_memory_allocated"],
            get_padding(model),
        ])

        h5f.close()
        return np.concatenate([stats, matched_scores, mismatched_scores])

    def routine(batch_sampler):
        for i, (dynamic, batch_type) in enumerate(zip(
            [False, True],
            ['seq.', 's'],
        )):

            scores = []

            if dynamic:
                batch_sizes = dynamic_sizes
            else:
                batch_sizes = fixed_sizes

            for batch_size in batch_sizes:
                scores_seeds = []
                for seed in args.seeds:
                    model = model_init.get_path_from_kwargs(
                        arch='convtasnet',
                        train_path=arg_type_path(p_train),
                        batch_size=float(batch_size),
                        batch_sampler=batch_sampler,
                        dynamic_batch_size=dynamic,
                        segment_length=segment_length,
                        seed=seed,
                    )
                    scores_seeds.append(load_score(model))
                    train_curver_plotter.plot(
                        ax, model, batch_size, batch_sampler, dynamic,
                    )
                scores.append(np.mean(scores_seeds, axis=0))
                train_curver_plotter.next_color()

            scores = np.array(scores)
            plot_models(scores, batch_sizes, batch_type)

            if i == 1:
                print(r'\hline \hline')
            else:
                print(r'\hhline{~==========}')

    # seeds = [0]
    fixed_sizes = [1, 2, 4, 8]
    dynamic_sizes = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    segment_length = 0.0

    print(r'\begin{table*}[t!]')
    print(r'\centering')
    print(r'\begin{tabular}{', end='')
    print(r'c', end='')
    print(r'r', end='')
    print(r'r', end='')
    print(r'r', end='')
    print(r'r', end='')
    print(r'c', end='')
    print(r'c', end='')
    print(r'c', end='')
    print(r'c', end='')
    print(r'c', end='')
    print(r'c', end='')
    print(r'}', end='')

    multicolspec = fr'\multicolumn{{{N_METRICS}}}{{c}}'
    print(r'\hline \hline')
    print(fr'&&&&&{multicolspec}{{Match}}&{multicolspec}{{Mismatch}}\\')
    print(r'Strategy')
    print(r'& \multicolumn{1}{c}{Batch size}')
    print(r'& \multicolumn{1}{c}{Time}')
    print(r'& \multicolumn{1}{c}{Memory}')
    print(r'& \multicolumn{1}{c}{Padding}')
    for _ in range(2):
        for m in METRICS:
            print(rf'& $\Delta${m["name"]}')
    print(r' \\ \hline \hline')

    print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Random}}')
    routine('random')
    print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Sorted}}')
    routine('sorted')
    print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Bucket}}')
    routine('bucket')

    print(r'\end{tabular}')
    print(r'\caption{Training statistics and scores in matched and mismatched '
          'conditions for different batching strategies.}')
    print(r'\label{tab:batching}')
    print(r'\end{table*}')
    print('')

    lines = [
        Line2D([], [], color='k', linestyle='-'),
        Line2D([], [], color='k', linestyle='--'),
    ]
    lh = ax.legend(loc=1)
    ax.legend(lines, ['train', 'val'], loc=2)
    ax.add_artist(lh)
    ax.grid()

    if args.show_plots:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--show-plots', action='store_true')
    args = parser.parse_args()
    main()
