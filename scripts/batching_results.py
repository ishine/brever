import itertools
import os
import json
import logging

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


databases = [
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


def fmt_time(time_):
    h, m, s = int(time_//3600), int((time_ % 3600)//60), int(time_ % 60)
    return f'{h:>2} h {m:>2} m {s:>2} s'


def fmt_memory(memory):
    memory = round(memory/1e9, 2)
    return f'{memory:>5} GB'


def fmt_score(score):
    return f"{score:.2e}"


def fmt_time_tex(time_):
    h, m = int(time_//3600), int((time_ % 3600)//60)
    return fr'{h} h {m:02d} m'


def fmt_score_tex(score, is_max):
    score = f"{score:.2f}"
    if is_max:
        score = fr"\textbf{{{score}}}"
    return score


def fmt_memory_tex(memory):
    memory = round(memory/1e9, 1)
    return f'{memory} GB'


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
    # val_batch_sampler = batch_sampler_class(
    #     dataset=val_split,
    #     **kwargs,
    # )

    batch_sizes, pad_amounts = train_batch_sampler.calc_batch_stats()
    percent = sum(pad_amounts)/(sum(batch_sizes)-sum(pad_amounts))*100
    percent = f'{round(percent, 1)}\\%'
    return percent


def complement(idx_list):
    return [i for i in range(5) if i not in idx_list]


def build_test_index(index, dims):
    test_index = [complement(index[dim]) for dim in range(3)]
    for dim in dims:
        test_index[dim] = index[dim]
    return test_index


def main():
    dset_init = DatasetInitializer(batch_mode=True)
    model_init = ModelInitializer(batch_mode=True)

    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
    plt.rc('grid', color='w', linestyle='solid')

    summary = {}

    fig, ax = plt.subplots(figsize=(4, 3))

    fig_test, axes_test = plt.subplots(1, 3, figsize=(5, 1.5))
    axes_test[0].set_title('PESQi')
    axes_test[1].set_title('STOIi')
    axes_test[2].set_title('SNRi')

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
    # plt.show()

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

    def plot_models(models, scores, batch_sizes, batch_type):
        max_score = np.nanmax(scores, axis=0)
        for model, score, batch_size in zip(models, scores, batch_sizes):
            plot_model(model, score, batch_size, max_score, batch_type)

    def plot_model(model, score, batch_size, max_score, batch_type):
        path = os.path.join(model, 'losses.npz')
        if not os.path.exists(path):
            logging.warning(f"{path} not found, skipping")
            return
        data = np.load(path)

        summary[model] = {}

        label = model

        l, = ax.plot(data['train'], label=label)
        _, = ax.plot(data['val'], '--', color=l.get_color())

        state = torch.load(os.path.join(model, 'checkpoint.pt'),
                           map_location='cpu')
        summary[model]['Training time'] = fmt_time(state['time_spent'])
        summary[model]['GPU usage'] = fmt_memory(state['max_memory_allocated'])
        summary[model]['Min. val. loss'] = fmt_score(min(data['val']))

        i_model = len(summary)
        pesq, stoi, snr, pesq_mis, stoi_mis, snr_mis = score
        is_max = score == max_score

        summary[model]['Test PESQi'] = fmt_score(pesq)
        summary[model]['Test STOIi'] = fmt_score(stoi)
        summary[model]['Test SNRi'] = fmt_score(snr)

        axes_test[0].bar([i_model], [pesq], width=1, label=label)
        axes_test[1].bar([i_model], [stoi], width=1, label=label)
        axes_test[2].bar([i_model], [snr], width=1, label=label)

        if batch_type == 'seq.':
            batch_size = int(batch_size)
        batch_size = f'{batch_size} {batch_type}'
        print(fr' & {batch_size}', end='')
        print(fr' & {fmt_time_tex(state["time_spent"])}', end='')
        print(fr' & {fmt_memory_tex(state["max_memory_allocated"])}', end='')
        print(fr' & {get_padding(model)}', end='')
        print(fr' & {fmt_score_tex(pesq*10, is_max[0])}', end='')
        print(fr' & {fmt_score_tex(stoi*100, is_max[1])}', end='')
        print(fr' & {fmt_score_tex(snr, is_max[2])}', end='')
        print(fr' & {fmt_score_tex(pesq_mis*10, is_max[3])}', end='')
        print(fr' & {fmt_score_tex(stoi_mis*100, is_max[4])}', end='')
        print(fr' & {fmt_score_tex(snr_mis, is_max[5])}', end='')
        print(r'\\')

    def get_test_dsets(index):
        test_paths = []
        for i, j, k in itertools.product(*index):
            test_path = dset_init.get_path_from_kwargs(
                kind='test',
                speakers={databases[0]['databases'][i]},
                noises={databases[1]['databases'][j]},
                rooms={databases[2]['databases'][k]},
                speech_files=[0.8, 1.0],
                noise_files=[0.8, 1.0],
                room_files='odd',
                duration=3600,
                seed=42,
            )
            test_paths.append(test_path)
        return test_paths

    def load_score(model):
        if not os.path.exists(model):
            raise FileNotFoundError(f'model {model} does not exist')
        score_file = os.path.join(model, 'scores.json')
        if not os.path.exists(score_file):
            return [np.nan for _ in range(6)]
        with open(score_file) as f:
            scores = json.load(f)
        if p_test not in scores.keys():
            logging.warning(f'model {model} is not tested on {p_test}')
            return [np.nan for _ in range(6)]
        pesq = np.mean(scores[p_test]['model']['PESQ'])
        stoi = np.mean(scores[p_test]['model']['STOI'])
        snr = np.mean(scores[p_test]['model']['SNR'])
        pesq -= np.mean(scores[p_test]['ref']['PESQ'])
        stoi -= np.mean(scores[p_test]['ref']['STOI'])
        snr -= np.mean(scores[p_test]['ref']['SNR'])
        matched = [pesq, stoi, snr]

        train_index = [[1], [0], [0]]
        test_idx = build_test_index(train_index, [])
        test_paths = get_test_dsets(test_idx)
        mismatched = []
        for test_path in test_paths:
            if test_path not in scores.keys():
                logging.warning(f'model {model} is not tested on {test_path}')
                return [np.nan for _ in range(6)]
            pesq = np.mean(scores[test_path]['model']['PESQ'])
            stoi = np.mean(scores[test_path]['model']['STOI'])
            snr = np.mean(scores[test_path]['model']['SNR'])
            pesq -= np.mean(scores[test_path]['ref']['PESQ'])
            stoi -= np.mean(scores[test_path]['ref']['STOI'])
            snr -= np.mean(scores[test_path]['ref']['SNR'])
            mismatched.append([pesq, stoi, snr])
        mismatched = np.array(mismatched).mean(axis=0)

        return matched + mismatched.tolist()

    def routine(batch_sampler):
        for i, (dynamic, batch_type) in enumerate(zip(
            [False, True],
            ['seq.', 's'],
        )):

            models = []
            scores = []

            if dynamic:
                batch_sizes = dynamic_sizes
            else:
                batch_sizes = fixed_sizes

            for batch_size in batch_sizes:

                m = model_init.get_path_from_kwargs(
                    arch='convtasnet',
                    train_path=arg_type_path(p_train),
                    batch_size=float(batch_size),
                    batch_sampler=batch_sampler,
                    dynamic_batch_size=dynamic,
                    segment_length=segment_length,
                    # seed=seed,
                )
                models.append(m)
                scores.append(load_score(m))

            scores = np.array(scores)
            plot_models(models, scores, batch_sizes, batch_type)

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

    print(r'\hline \hline')
    print(r'&&&&&\multicolumn{3}{c}{Match}&\multicolumn{3}{c}{Mismatch}\\')
    print(r'Strategy')
    print(r'& \multicolumn{1}{c}{Batch size}')
    print(r'& \multicolumn{1}{c}{Time}')
    print(r'& \multicolumn{1}{c}{Memory}')
    print(r'& \multicolumn{1}{c}{Padding}')
    print(r'& $\Delta$PESQ')
    print(r'& $\Delta$STOI')
    print(r'& $\Delta$SNR')
    print(r'& $\Delta$PESQ')
    print(r'& $\Delta$STOI')
    print(r'& $\Delta$SNR')
    print(r' \\ \hline \hline')

    print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Random}}')
    routine('random')
    print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Sorted}}')
    routine('sorted')
    print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Bucket}}')
    routine('bucket')

    print(r'\end{tabular}')
    print(r'\caption{Caption}')
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

    pretty_table(summary)

    fig_test.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()