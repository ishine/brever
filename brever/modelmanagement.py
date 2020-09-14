import os
import json
import pickle
import hashlib

import yaml

from .config import defaults


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
                    output[f'{prefix}.{key}'] = value
        else:
            if prefix is None:
                output[key] = value
            else:
                output[f'{prefix}.{key}'] = value
    return output


def unflatten(keys, values):
    output = []
    for item in values:
        config = {}
        for key, value in zip(keys, item):
            path = key.split('.')
            subdict = config
            for subkey in path[:-1]:
                if subkey not in subdict.keys():
                    subdict[subkey] = {}
                subdict = subdict[subkey]
            subdict[path[-1]] = value
        output.append(config)
    return output


def get_feature_indices(train_path, features):
    if isinstance(features, set):
        features = sorted(features)
    pipes_path = os.path.join(train_path, 'pipes.pkl')
    with open(pipes_path, 'rb') as f:
        featureExtractor = pickle.load(f)['featureExtractor']
    names = featureExtractor.features
    indices = featureExtractor.indices
    indices_dict = {name: lims for name, lims in zip(names, indices)}
    if 'itd_ic' in indices_dict.keys():
        start, end = indices_dict.pop('itd_ic')
        step = (end - start)//2
        indices_dict['itd'] = (start, start+step)
        indices_dict['ic'] = (start+step, end)
    feature_indices = [indices_dict[feature] for feature in features]
    return feature_indices


def get_file_indices(train_path):
    metadatas_path = os.path.join(train_path, 'mixture_info.json')
    with open(metadatas_path, 'r') as f:
        metadatas = json.load(f)
        indices = [item['dataset_indices'] for item in metadatas]
    return indices


def set_dict_field(input_dict, key_list, value):
    dict_ = input_dict
    for key in key_list:
        if key not in dict_.keys():
            dict_[key] = {}
        if key == key_list[-1]:
            dict_[key] = value
        else:
            dict_ = dict_[key]


def get_dict_field(input_dict, key_list, default=None):
    try:
        dict_ = input_dict
        for key in key_list:
            if key == key_list[-1]:
                return dict_[key]
            else:
                dict_ = dict_[key]
    except KeyError:
        return default


def find_model(**kwargs):
    models = []
    for model_id in os.listdir('models'):
        config_file = os.path.join('models', model_id, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        valid = True
        for attr, keys in [
                    ('layers', ['MODEL', 'NLAYERS']),
                    ('stacks', ['POST', 'STACK']),
                    ('batchnorm', ['MODEL', 'BATCHNORM', 'ON']),
                    ('dropout', ['MODEL', 'DROPOUT', 'ON']),
                    ('batchsize', ['MODEL', 'BATCHSIZE']),
                    ('features', ['POST', 'FEATURES']),
                    ('train_path', ['POST', 'PATH', 'TRAIN']),
                    ('val_path', ['POST', 'PATH', 'VAL']),
                ]:
            if attr in kwargs.keys():
                value = kwargs[attr]
                if value is not None and get_dict_field(config, keys) != value:
                    valid = False
                    break
        if valid:
            models.append(model_id)
    return models
