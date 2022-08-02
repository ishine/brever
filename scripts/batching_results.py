import itertools
import os
import json

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


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


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
        segment_length=config.TRAINING.SEGMENT_LENGTH,
        sorted_=config.TRAINING.BATCH_SAMPLER.SORTED,
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

    fig, ax = plt.subplots()

    fig_test, axes_test = plt.subplots(1, 3, figsize=(10, 3))
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

    def plot_models(models, scores, kw_list, batch_type):
        max_score = np.max(scores, axis=0)
        for model, score, kwargs in zip(models, scores, kw_list):
            plot_model(model, score, kwargs, max_score, batch_type)

    def plot_model(model, score, kwargs, max_score, batch_type):
        path = os.path.join(model, 'losses.npz')
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
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

        if "sort_observations" not in kwargs.keys():
            kwargs["sort_observations"] = False
        batch_size = kwargs["batch_size"]
        if batch_type == 'seq.':
            batch_size = int(kwargs["batch_size"])
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
        score_file = os.path.join(model, 'scores.json')
        with open(score_file) as f:
            scores = json.load(f)
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
            pesq = np.mean(scores[test_path]['model']['PESQ'])
            stoi = np.mean(scores[test_path]['model']['STOI'])
            snr = np.mean(scores[test_path]['model']['SNR'])
            pesq -= np.mean(scores[test_path]['ref']['PESQ'])
            stoi -= np.mean(scores[test_path]['ref']['STOI'])
            snr -= np.mean(scores[test_path]['ref']['SNR'])
            mismatched.append([pesq, stoi, snr])
        mismatched = np.array(mismatched).mean(axis=0)

        return matched + mismatched.tolist()

    def routine(batch_type, **kwargs):
        # basic batch samplers
        models = []
        scores = []
        kw_list = []
        for kwargs in product_dict(**kwargs):
            m = model_init.get_path_from_kwargs(
                arch='convtasnet',
                train_path=arg_type_path(p_train),
                **kwargs,
            )
            models.append(m)
            scores.append(load_score(m))
            kw_list.append(kwargs)
        scores = np.array(scores)
        plot_models(models, scores, kw_list, batch_type)

    print(r'\begin{table*}')
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

    print(r'\multirow{12}{*}{\rotatebox[origin=c]{90}{Random}}')
    routine(
        batch_type='seq.',
        sort_observations=[False],
        batch_size=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        batch_sampler=['simple']
    )
    print(r'\hhline{~==========}')
    routine(
        batch_type='s.',
        sort_observations=[False],
        batch_size=[4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        batch_sampler=['dynamic']
    )
    print(r'\hline \hline')
    print(r'\multirow{12}{*}{\rotatebox[origin=c]{90}{Sorted}}')
    routine(
        batch_type='seq.',
        sort_observations=[True],
        batch_size=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        batch_sampler=['simple']
    )
    print(r'\hhline{~==========}')
    routine(
        batch_type='s',
        sort_observations=[True],
        batch_size=[4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        batch_sampler=['dynamic']
    )
    print(r'\hline \hline')
    print(r'\multirow{6}{*}{\rotatebox[origin=c]{90}{Bucket}}')
    routine(
        batch_type='s',
        batch_size=[4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        batch_sampler=['bucket']
    )
    print(r'\hline \hline')

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
