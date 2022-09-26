import itertools
import os
import h5py
import argparse

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.stats

from brever.args import arg_type_path
from brever.config import DatasetInitializer, ModelInitializer, get_config
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

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
plt.rc('grid', color='w', linestyle='solid')

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
        'name': 'ESTOI',
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


def fmt_score(score, is_max=False):
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


def get_dataset_paths(dset_init):
    # train dataset
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
    # matched test dataset
    p_match = dset_init.get_path_from_kwargs(
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
    # mismatched test dataset
    train_index = [[1], [0], [0]]
    test_idx = build_test_index(train_index, [])
    p_mismatch = []
    for i, j, k in itertools.product(*test_idx):
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
        p_mismatch.append(test_path)
    return p_train, p_match, p_mismatch


def plot_distribution(path):
    dataset = BreverDataset(
        path=path,
        segment_length=0.0,
        fs=16000,
    )
    lengths = np.array(dataset.item_lengths)/16e3
    plt.figure(figsize=(3.5, 1))
    plt.hist(lengths, bins=25)
    plt.xlabel('Mixture length (s)')
    plt.ylabel('Count')
    plt.grid()
    if args.dist_fig_path is not None:
        plt.savefig(args.dist_fig_path, bbox_inches='tight', pad_inches=0)


class ScoreLoader:
    def __init__(self, p_match, p_mismatch):
        self.p_match = p_match
        self.p_mismatch = p_mismatch
        self._model = None
        self._file = None
        self._metric_idx = None

    def _open_file(self, model):
        if not os.path.exists(model):
            raise FileNotFoundError(f'model {model} does not exist')
        filename = os.path.join(model, 'scores.hdf5')
        file = h5py.File(filename)
        self._model = model
        self._file = file
        self._metric_idx = [
            list(file['metrics'].asstr()).index(m['name']) for m in METRICS
        ]

    def _close_file(self):
        self._file.close()
        self._model = None
        self._file = None
        self._metric_idx = None

    def _load(self, path):
        if path not in self._file.keys():
            raise KeyError(f'model {self._model} is not tested on {path}')
        scores = self._file[path][:, self._metric_idx, :].mean(axis=0)
        scores = scores[:, 1] - scores[:, 0]
        return scores

    def _load_match_score(self):
        return self._load(self.p_match)

    def _load_mismatch_score(self):
        output = []
        for p in self.p_mismatch:
            output.append(self._load(p))
        output = np.mean(output, axis=0)
        return output

    def _load_training_stats(self):
        state = torch.load(os.path.join(self._model, 'checkpoint.pt'),
                           map_location='cpu')
        output = np.array([
            state["time_spent"],
            state["max_memory_allocated"],
            get_padding(self._model),
        ])
        return output

    def load(self, model):
        self._open_file(model)
        match_scores = self._load_match_score()
        mismatch_scores = self._load_mismatch_score()
        training_stats = self._load_training_stats()
        self._close_file()
        return np.concatenate([training_stats, match_scores, mismatch_scores])


class TrainCurvePlotter:
    def __init__(self, ax):
        self.ax = ax
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.counter = 0

    def plot(self, model, **kwargs):
        path = os.path.join(model, 'losses.npz')
        data = np.load(path)
        label = ', '.join(f'{key}={val}' for key, val in kwargs.items())
        color = self.colors[self.counter]
        l, = self.ax.plot(data['train'], label=label, color=color)
        _, = self.ax.plot(data['val'], '--', color=color)

    def next_color(self):
        self.counter = (self.counter + 1) % len(self.colors)

    def add_legend(self):
        lines = [
            Line2D([], [], color='k', linestyle='-'),
            Line2D([], [], color='k', linestyle='--'),
        ]
        lh = self.ax.legend(loc=1)
        self.ax.legend(lines, ['train', 'val'], loc=2)
        self.ax.add_artist(lh)
        self.ax.grid()


class ModelGetter:
    def __init__(self, model_init, p_train, segment_length):
        self.model_init = model_init
        self.p_train = p_train
        self.segment_length = segment_length

    def get(self, batch_sampler, dynamic, batch_size, seed):
        return self.model_init.get_path_from_kwargs(
            arch='convtasnet',
            train_path=arg_type_path(self.p_train),
            batch_size=float(batch_size),
            batch_sampler=batch_sampler,
            dynamic_batch_size=dynamic,
            segment_length=self.segment_length,
            seed=seed,
        )


class SubSectionPrinter:
    def __init__(self, scores, sems, batch_sizes, dynamic):
        self.stats, self.scores = scores[:, :3], scores[:, 3:]
        self.sems = sems[:, 3:]
        self.max_score = np.max(self.scores, axis=0)
        self.max_sem = np.max(self.sems, axis=0)
        self.batch_sizes = batch_sizes
        self.dynamic = dynamic

    def print(self):
        for stat, score, sem, batch_size in zip(
            self.stats, self.scores, self.sems, self.batch_sizes
        ):
            self._print_model(stat, score, sem, batch_size)

    def _print_model(self, stat, score, sem, batch_size):
        is_max_score = score == self.max_score
        is_max_sem = sem == self.max_sem
        if self.dynamic:
            batch_unit = 's'
        else:
            batch_unit = 'seq.'
            batch_size = int(batch_size)
        batch_size = f'{batch_size} {batch_unit}'
        print(fr'& {batch_size}', end=' ')
        print(fr'& {fmt_time(stat[0])}', end=' ')
        print(fr'& {fmt_memory(stat[1])}', end=' ')
        print(fr'& {fmt_padding(stat[2])}', end=' ')
        for i_offset in [0, 3]:  # allows to grab matched or mismatched score
            for i_m, m in enumerate(METRICS):
                i_s = i_m + i_offset
                score_fmt = fmt_score(score[i_s]*m["scale"], is_max_score[i_s])
                if args.print_sem:
                    sem_fmt = fmt_score(sem[i_s]*m["scale"], is_max_sem[i_s])
                    sem_fmt = rf' {{\scriptsize $\pm$ {sem_fmt}}}'
                    score_fmt += sem_fmt
                print(fr'& {score_fmt}', end=' ')
        print(r'\\')


class SeedLoader:
    def __init__(self, model_getter, score_loader, train_curve_plotter):
        self.model_getter = model_getter
        self.score_loader = score_loader
        self.train_curve_plotter = train_curve_plotter
        self.params = None

    def set_params(self, **kwargs):
        self.params = kwargs

    def load_seed(self, seed):
        model = self.model_getter.get(seed=seed, **self.params)
        score = self.score_loader.load(model)
        self.train_curve_plotter.plot(
            model, **self.params,
        )
        return score

    def load_seeds(self, seeds, return_sem=False):
        scores = [self.load_seed(seed) for seed in seeds]
        mean = np.mean(scores, axis=0)
        if return_sem:
            return mean, scipy.stats.sem(scores, axis=0)
        else:
            return mean


class SectionPrinter:
    def __init__(self, train_curve_plotter, seed_loader, fixed_sizes,
                 dynamic_sizes):
        self.train_curve_plotter = train_curve_plotter
        self.seed_loader = seed_loader
        self.fixed_sizes = fixed_sizes
        self.dynamic_sizes = dynamic_sizes

    def _print_subsection(self, batch_sampler, dynamic):
        if dynamic:
            batch_sizes = self.dynamic_sizes
        else:
            batch_sizes = self.fixed_sizes

        scores = []
        sems = []
        for batch_size in batch_sizes:
            self.seed_loader.set_params(
                batch_sampler=batch_sampler,
                dynamic=dynamic,
                batch_size=batch_size,
            )
            mean, sem = self.seed_loader.load_seeds(args.seeds, True)
            scores.append(mean)
            sems.append(sem)
            self.train_curve_plotter.next_color()

        scores = np.array(scores)
        sems = np.array(sems)

        printer = SubSectionPrinter(scores, sems, batch_sizes, dynamic)
        printer.print()

    def print_section(self, batch_sampler):
        for i, dynamic in enumerate([False, True]):
            self._print_subsection(batch_sampler, dynamic)
            if i == 1:
                print(r'\hline \hline')
            else:
                print(r'\hhline{~==========}')

    def print_multirow(self, n, header):
        print(rf'\multirow{{{n}}}{{*}}{{\rotatebox[origin=c]{{90}}'
              rf'{{{header}}}}}')

    def print_table_start(self):
        colspec = f'crrrr{"c"*N_METRICS*2}'
        print(r'\begin{table*}[t!]')
        print(r'\centering')
        print(rf'\begin{{tabular}}{{{colspec}}}')
        print(r'\hline \hline')

    def print_first_row(self):
        multicolspec = fr'\multicolumn{{{N_METRICS}}}{{c}}'
        print(fr'&&&&&{multicolspec}{{Match}}&{multicolspec}{{Mismatch}}\\')
        cols = [
            r'Strategy',
            r'\multicolumn{1}{c}{Batch size}',
            r'\multicolumn{1}{c}{Time}',
            r'\multicolumn{1}{c}{Memory}',
            r'\multicolumn{1}{c}{Padding}',
        ]
        cols += [rf'$\Delta${m["name"]}' for _ in range(2) for m in METRICS]
        print(' & '.join(cols) + r' \\')
        print(r'\hline \hline')

    def print_table_end(self, caption, label):
        print(r'\end{tabular}')
        print(rf'\caption{{{caption}}}')
        print(rf'\label{{{label}}}')
        print(r'\end{table*}')
        print('')


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    fig, ax = plt.subplots(figsize=(4, 3))

    fixed_sizes = [1, 2, 4, 8]
    dynamic_sizes = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    segment_length = 0.0

    p_train, p_match, p_mismatch = get_dataset_paths(dset_init)
    score_loader = ScoreLoader(p_match, p_mismatch)
    model_getter = ModelGetter(model_init, p_train, segment_length)
    plotter = TrainCurvePlotter(ax)
    seed_loader = SeedLoader(model_getter, score_loader, plotter)
    printer = SectionPrinter(plotter, seed_loader, fixed_sizes, dynamic_sizes)

    caption = (
        'Training statistics and scores in matched and mismatched conditions '
        'for different batching strategies.'
    )
    label = 'tab:batching'

    printer.print_table_start()
    printer.print_first_row()
    printer.print_multirow(10, 'Random')
    printer.print_section('random')
    printer.print_multirow(10, 'Sorted')
    printer.print_section('sorted')
    printer.print_multirow(10, 'Bucket')
    printer.print_section('bucket')
    printer.print_table_end(caption, label)

    plotter.add_legend()

    if args.show_plots:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--print-sem', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--dist-fig-path', type=str)
    args = parser.parse_args()
    main()
