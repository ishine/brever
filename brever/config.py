import hashlib
import os

import yaml

from .args import ModelArgParser, DatasetArgParser


def get_config(path):
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    config = BreverConfig(config_dict)
    return config


class BreverConfig:
    def __init__(self, dict_):
        for key, value in dict_.items():
            if isinstance(value, dict):
                super().__setattr__(key, BreverConfig(value))
            else:
                super().__setattr__(key, value)

    def __setattr__(self, attr, value):
        class_name = self.__class__.__name__
        raise AttributeError(f'{class_name} objects are immutable')

    def to_dict(self):
        dict_ = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BreverConfig):
                dict_[key] = value.to_dict()
            else:
                dict_[key] = value
        return dict_

    def get_hash(self, length=8):

        def sorted_dict(input_dict):
            output_dict = {}
            for key, value in sorted(input_dict.items()):
                if isinstance(value, dict):
                    output_dict[key] = sorted_dict(value)
                elif isinstance(value, set):
                    output_dict[key] = sorted(value)
                else:
                    output_dict[key] = value
            return output_dict

        dict_ = self.to_dict()
        dict_ = sorted_dict(dict_)
        str_ = str(dict_.items())
        hash_ = hashlib.sha256(str_.encode()).hexdigest()
        return hash_[:length]

    def get_field(self, key_list):
        attr = getattr(self, key_list[0])
        if len(key_list) == 1:
            return attr
        else:
            return attr.get_field(key_list[1:])

    def set_field(self, key_list, value):
        if len(key_list) == 1:
            key = key_list[0]
            attr = getattr(self, key)
            if not isinstance(value, type(attr)):
                type_a = attr.__class__.__name__
                type_v = value.__class__.__name__
                msg = f'attribute {key} must be {type_a}, got {type_v}'
                raise TypeError(msg)
            super().__setattr__(key, value)
        else:
            config = self.get_field(key_list[:-1])
            config.set_field(key_list[-1:], value)

    def update_from_args(self, args, arg_map):
        for arg_name, key_list in arg_map.items():
            value = getattr(args, arg_name)
            if value is not None:
                self.set_field(key_list, value)

    def update_from_dict(self, dict_, parent_keys=[]):

        def flatten_dict(dict_, parent_keys=[]):
            for key, value in dict_.items():
                key_list = parent_keys + [key]
                if isinstance(value, dict):
                    yield from flatten_dict(value, key_list)
                else:
                    yield key_list, value

        for key_list, value in flatten_dict(dict_):
            self.set_field(key_list, value)


class ModelFinder:
    def __init__(self):
        self.models = None
        self.configs = None

    def find(self, **kwargs):
        if self.models is None:
            self.models = []
            paths = get_config('config/paths.yaml')
            models_dir = paths.MODELS
            for model_id in os.listdir(models_dir):
                self.models.append(os.path.join(models_dir, model_id))

        if self.configs is None:
            self.configs = None
            for model in self.models:
                config_file = os.path.join(model, 'config.yaml')
                config = get_config(config_file)
                self.configs.append(config)

        assert len(self.models) == len(self.configs)

        models = []
        for model, config in zip(self.models):
            valid = True
            for key, values in kwargs.items():
                key_list = ModelArgParser.arg_map['arch'][key]
                if config.get_field(key_list) not in values:
                    valid = False
                    break
            if valid:
                models.append(model)

        return models


class DatasetFinder:
    def __init__(self):
        self.dsets = None
        self.configs = None

    def find(self, kind=None, **kwargs):
        if self.dsets is None:
            self.dsets = []
            paths = get_config('config/paths.yaml')
            dsets_dir = paths.DATASETS
            if kind is not None:
                directory = os.path.join(dsets_dir, kind)
            for root, folder, files in os.walk(directory):
                if 'config.yaml' in files:
                    self.dsets.append(root)

        if self.configs is None:
            self.configs = []
            for dset in self.dsets:
                config_file = os.path.join(dset, 'config.yaml')
                config = get_config(config_file)
                self.configs.append(config)

        assert len(self.models) == len(self.configs)

        dsets = []
        for dset, config in zip(self.dsets, self.configs):
            valid = True
            for key, values in kwargs.items():
                key_list = DatasetArgParser.arg_map[key]
                if config.get_field(config, key_list) not in values:
                    valid = False
                    break
            if valid:
                dsets.append(dset.replace('\\', '/'))

        return dsets
