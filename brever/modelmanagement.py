import os
import json
import pickle
import hashlib
import argparse

import yaml

from .config import defaults


def sorted_dict(data, config=None):
    if config is None:
        config = defaults()
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


arg_to_keys_map = {
    'layers': ['MODEL', 'NLAYERS'],
    'stacks': ['POST', 'STACK'],
    'batchnorm': ['MODEL', 'BATCHNORM', 'ON'],
    'dropout': ['MODEL', 'DROPOUT', 'ON'],
    'dropout_input': ['MODEL', 'DROPOUT', 'INPUT'],
    'batchsize': ['MODEL', 'BATCHSIZE'],
    'features': ['POST', 'FEATURES'],
    'train_path': ['POST', 'PATH', 'TRAIN'],
    'val_path': ['POST', 'PATH', 'VAL'],
    'dct': ['POST', 'DCT', 'ON'],
    'n_dct': ['POST', 'DCT', 'NCOEFF'],
    'cuda': ['MODEL', 'CUDA'],
}


def find_model(**kwargs):
    models = []
    for model_id in os.listdir('models'):
        config_file = os.path.join('models', model_id, 'config_full.yaml')
        if not os.path.exists(config_file):
            config_file = os.path.join('models', model_id, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        valid = True
        for key, value in kwargs.items():
            keys = arg_to_keys_map[key]
            if value is not None and get_dict_field(config, keys) not in value:
                valid = False
                break
        if valid:
            models.append(model_id)
    return models


class ExtendableArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_dests = []
        self._extra_dests = []

    def add_base_argument(self, *args, **kwargs):
        _storeAction = super().add_argument(*args, **kwargs)
        self._base_dests.append(_storeAction.dest)

    def add_argument(self, *args, **kwargs):
        _storeAction = super().add_argument(*args, **kwargs)
        if args == ('-h', '--help'):
            return
        self._extra_dests.append(_storeAction.dest)

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        base_args = argparse.Namespace()
        extra_args = argparse.Namespace()
        for dest in self._base_dests:
            setattr(base_args, dest, getattr(args, dest))
        for dest in self._extra_dests:
            setattr(extra_args, dest, getattr(args, dest))
        return base_args, extra_args


class ModelFilterArgParser(ExtendableArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_base_argument(
            '--layers',
            type=int,
            nargs='+',
            help='number of layers',
        )
        self.add_base_argument(
            '--stacks',
            type=int,
            nargs='+',
            help='number of extra stacks',
        )
        self.add_base_argument(
            '--batchnorm',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='batchnorm toggle',
        )
        self.add_base_argument(
            '--dropout',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='dropout toggle',
        )
        self.add_base_argument(
            '--batchsize',
            type=int,
            nargs='+',
            help='batchsize',
        )
        self.add_base_argument(
            '--features',
            type=lambda x: set(x.split(' ')),
            nargs='+',
            help='feature set',
        )
        self.add_base_argument(
            '--train-path',
            type=lambda x: x.replace('\\', '/'),
            nargs='+',
            help='training dataset path',
        )
        self.add_base_argument(
            '--val-path',
            type=lambda x: x.replace('\\', '/'),
            nargs='+',
            help='validation dataset path',
        )
        self.add_base_argument(
            '--dropout-input',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='dropout input layer toggle',
        )
        self.add_base_argument(
            '--n-dct',
            type=int,
            nargs='+',
            help='number of dct coefficients',
        )
        self.add_base_argument(
            '--dct',
            type=int,
            nargs='+',
            help='dct toggle',
        )


class DatasetInitArgParser(ExtendableArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_base_argument(
            '--decay',
            type=lambda x: bool(int(x)),
            help='decaying noise toggle',
        )
        self.add_base_argument(
            '--decay-color',
            type=str,
            help='decaying noise color',
        )
        self.add_base_argument(
            '--diffuse',
            type=lambda x: bool(int(x)),
            help='diffuse noise toggle',
        )
        self.add_base_argument(
            '--diffuse-color',
            type=str,
            help='diffuse noise color',
        )
        self.add_base_argument(
            '--drr-min',
            type=int,
            help='random decay drr lower bound',
        )
        self.add_base_argument(
            '--drr-max',
            type=int,
            help='random decay drr upper bound',
        )
        self.add_base_argument(
            '--rt60-min',
            type=float,
            help='random decay rt60 lower bound',
        )
        self.add_base_argument(
            '--rt60-max',
            type=float,
            help='random decay rt60 upper bound',
        )
        self.add_base_argument(
            '--delay-min',
            type=float,
            help='random decay delay lower bound',
        )
        self.add_base_argument(
            '--delay-max',
            type=float,
            help='random decay delay upper bound',
        )
        self.add_base_argument(
            '--rooms',
            nargs='+',
            type=str,
            help='list of rooms',
        )

    def parse_args(self, *args, **kwargs):
        base_args, extra_args = super().parse_args(*args, **kwargs)
        if base_args.rooms is not None:
            base_args.rooms = set(base_args.rooms)
        return base_args, extra_args
