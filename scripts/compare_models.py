import os

import numpy as np
import matplotlib.pyplot as plt
import yaml

from brever.modelmanagement import (get_dict_field, ModelFilterArgParser,
                                    find_model, arg_to_keys_map)


def check_models(models, dims):
    values = []
    models_ = []
    for model in models:
        pesq_file = os.path.join('models', model, 'eval_PESQ.npy')
        mse_file = os.path.join('models', model, 'eval_MSE.npy')
        config_file = os.path.join('models', model, 'config_full.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if not (os.path.exists(pesq_file) and os.path.exists(mse_file)):
            print(f'Model {model} is not evaluated!')
            continue
        val = {dim: get_dict_field(config, arg_to_keys_map[dim])
               for dim in dims}
        if val not in values:
            values.append(val)
            models_.append(model)
        else:
            raise ValueError(f'Found more than one model for value {val}')
    return models_, values


def group_by_dimension(models, values, dimension):
    # first make groups
    if dimension is None:
        group = []
        for model, val in zip(models, values):
            group.append({'model': model, 'val': val})
        groups = [group]
    else:
        group_outer_values = []
        groups = []
        group_inner_values = []
        for model, val in zip(models, values):
            group_outer_val = val[dimension]
            if group_outer_val not in group_outer_values:
                group_outer_values.append(group_outer_val)
                groups.append([{'model': model, 'val': val}])
            else:
                index = group_outer_values.index(group_outer_val)
                groups[index].append({'model': model, 'val': val})
            group_inner_val = val.copy()
            group_inner_val.pop(dimension)
            if group_inner_val not in group_inner_values:
                group_inner_values.append(group_inner_val)
    # then match order across groups
    for i, group in enumerate(groups):
        group_sorted = []
        group_inner_vals_local = [model['val'].copy() for model in group]
        for val in group_inner_vals_local:
            val.pop(dimension)
        for group_inner_val in group_inner_values:
            if group_inner_val in group_inner_vals_local:
                index = group_inner_vals_local.index(group_inner_val)
                group_sorted.append(group[index])
        groups[i] = group_sorted
    return groups


def load_pesq_and_mse(groups):
    for group in groups:
        for i in range(len(group)):
            model = group[i]['model']
            pesq = np.load(os.path.join('models', model, 'eval_PESQ.npy'))
            mse = np.load(os.path.join('models', model, 'eval_MSE.npy'))
            group[i]['pesq'] = pesq
            group[i]['mse'] = mse


def sort_groups_by_mean_pesq(groups):
    groups_mean_pesq = []
    for group in groups:
        mean_pesqs = [model['pesq'].mean() for model in group]
        group_mean_pesq = np.mean(mean_pesqs)
        groups_mean_pesq.append(group_mean_pesq)
    indexes = np.argsort(groups_mean_pesq)
    groups = [groups[i] for i in indexes]
    return groups


def main(dimensions, group_by, filter_):
    models = find_model(**filter_)
    models, values = check_models(models, dimensions)
    groups = group_by_dimension(models, values, group_by)
    load_pesq_and_mse(groups)
    groups = sort_groups_by_mean_pesq(groups)

    try:
        i_values_sorted = np.argsort(values)
    except TypeError:
        i_values_sorted = np.arange(len(values))

    print(f'Comparing {len(models)} models:')
    for i in i_values_sorted:
        print(f'Model {models[i]} with dimension value {values[i]}')

    snrs = [0, 3, 6, 9, 12, 15]
    room_names = ['A', 'B', 'C', 'D']

    n = len(models)
    width = 1/(n+1)

    for ylabel, metric in zip(
                ['MSE', r'$\Delta PESQ$'],
                ['mse', 'pesq'],
            ):
        fig, axes = plt.subplots(1, 2, sharey=True)
        for axis, (ax, xticklabels, xlabel) in enumerate(zip(
                    axes[::-1],
                    [room_names, snrs],
                    ['Room', 'SNR (dB)'],
                )):
            model_count = 0
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            hatch_cycle = ['', '//', '////']
            for i, group in enumerate(groups):
                for j, model in enumerate(group):
                    data = model[metric].mean(axis=axis)
                    data = np.hstack((data, data.mean()))
                    label = f'{model["val"]}'
                    x = np.arange(len(data)) + (model_count - (n-1)/2)*width
                    x[-1] = x[-1] + 2*width
                    ax.bar(
                        x=x,
                        height=data,
                        width=width,
                        label=label,
                        color=color_cycle[i % len(color_cycle)],
                        hatch=hatch_cycle[j % len(hatch_cycle)],
                    )
                    model_count += 1
            xticks = np.arange(len(xticklabels) + 1, dtype=float)
            xticks[-1] = xticks[-1] + 2*width
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels + ['Mean'])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.yaxis.set_tick_params(labelleft=True)
        fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = ModelFilterArgParser(description='compare models')
    parser.add_argument('dimensions', nargs='+',
                        help='parameter dimensions to compare')
    parser.add_argument('--group-by',
                        help='parameter to group by')
    args = parser.parse_args()

    filter_ = vars(args).copy()
    filter_.pop('dimensions')
    filter_.pop('group_by')

    main(args.dimensions, args.group_by, filter_)
