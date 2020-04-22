import os
import json
import pickle
import hashlib

from brever.config import defaults


def sorted_dict(data, config=defaults()):
    output = {}
    for key, value in sorted(data.items()):
        if isinstance(value, dict):
            output[key] = sorted_dict(value, config=getattr(config, key))
        else:
            if isinstance(getattr(config, key), set):
                output[key] = sorted(value)
            else:
                output[key] = value
    return output


def get_unique_id(data):
    if not data:
        data = {}
    data = sorted_dict(data)
    unique_str = ''.join([f'{hashlib.sha256(str(key).encode()).hexdigest()}'
                          f'{hashlib.sha256(str(val).encode()).hexdigest()}'
                          for key, val in data.items()])
    unique_id = hashlib.sha256(unique_str.encode()).hexdigest()
    return unique_id


def flatten(dictionary, prefix=None):
    output = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            for key, value in flatten(value, prefix=key).items():
                if prefix is None:
                    output[key] = value
                else:
                    output[f'{prefix}_{key}'] = value
        else:
            if prefix is None:
                output[key] = value
            else:
                output[f'{prefix}_{key}'] = value
    return output


def unflatten(keys, values):
    output = []
    for item in values:
        config = {}
        for key, value in zip(keys, item):
            path = key.split('_')
            subdict = config
            for subkey in path[:-1]:
                if subkey not in subdict.keys():
                    subdict[subkey] = {}
                subdict = subdict[subkey]
            subdict[path[-1]] = value
        output.append(config)
    return output


def get_feature_indices(train_path, features):
    pipes_path = os.path.join(train_path, 'pipes.pkl')
    with open(pipes_path, 'rb') as f:
        featureExtractor = pickle.load(f)['featureExtractor']
    names = featureExtractor.features
    indices = featureExtractor.indices
    indices_dict = {name: lims for name, lims in zip(names, indices)}
    itd_ic_indices = indices_dict.pop('itd_ic')
    itd_ic_mid = (itd_ic_indices[0] + itd_ic_indices[1])//2
    indices_dict['itd'] = (itd_ic_indices[0], itd_ic_mid)
    indices_dict['ic'] = (itd_ic_mid, itd_ic_indices[1])
    feature_indices = [indices_dict[feature] for feature in features]
    return feature_indices


def get_file_indices(train_path):
    metadatas_path = os.path.join(train_path, 'mixture_info.json')
    with open(metadatas_path, 'r') as f:
        metadatas = json.load(f)
        indices = [item['dataset_indices'] for item in metadatas]
    return indices
