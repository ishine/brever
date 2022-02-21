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

    def find(self, arch=None, **kwargs):
        if self.models is None:
            self.models = []
            paths = get_config('config/paths.yaml')
            models_dir = paths.MODELS
            for model_id in os.listdir(models_dir):
                self.models.append(os.path.join(models_dir, model_id))

        if self.configs is None:
            self.configs = []
            for model in self.models:
                config_file = os.path.join(model, 'config.yaml')
                config = get_config(config_file)
                self.configs.append(config)

        assert len(self.models) == len(self.configs)

        models = []
        configs = []
        for model, config in zip(self.models, self.configs):
            valid = True
            if arch is not None and config.ARCH != arch:
                valid = False
            else:
                for key, value in kwargs.items():
                    key_list = ModelArgParser.arg_map[config.ARCH][key]
                    if config.get_field(key_list) != value:
                        valid = False
                        break
            if valid:
                models.append(model)
                configs.append(config)

        return models, configs

    def find_from_args(self, args):
        if args.arch is None:
            arg_map = ModelArgParser.training_args
        else:
            arg_map = ModelArgParser.arg_map[args.arch]
        kwargs = {}
        for key in arg_map.keys():
            val = getattr(args, key)
            if val is not None:
                kwargs[key] = val
        return self.find(args.arch, **kwargs)


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
                dsets_dir = os.path.join(dsets_dir, kind)
            for root, folder, files in os.walk(dsets_dir):
                if 'config.yaml' in files:
                    self.dsets.append(root)

        if self.configs is None:
            self.configs = []
            for dset in self.dsets:
                config_file = os.path.join(dset, 'config.yaml')
                config = get_config(config_file)
                self.configs.append(config)

        assert len(self.dsets) == len(self.configs)

        dsets = []
        configs = []
        for dset, config in zip(self.dsets, self.configs):
            valid = True
            for key, value in kwargs.items():
                key_list = DatasetArgParser.arg_map[key]
                if config.get_field(key_list) != value:
                    valid = False
                    break
            if valid:
                dsets.append(dset)
                configs.append(config)

        return dsets, configs

    def find_from_args(self, args):
        arg_map = DatasetArgParser.arg_map
        kwargs = {}
        for key in arg_map.keys():
            val = getattr(args, key)
            if val is not None:
                kwargs[key] = val
        return self.find(args.kind, **kwargs)


class ModelInitializer:
    def __init__(self):
        self.dir_ = get_config('config/paths.yaml').MODELS

    def init_from_args(self, args):
        config = get_config(f'config/models/{args.arch}.yaml')
        config.update_from_args(args, ModelArgParser.arg_map[args.arch])
        return self.write_config(config, args.force)

    def init_from_kwargs(self, arch, force=False, **kwargs):
        config = get_config(f'config/models/{arch}.yaml')
        for key, val in kwargs.items():
            config.set_field(ModelArgParser.arg_map[arch][key], val)
        return self.write_config(config, force=force)

    def write_config(self, config, force=False):
        model_id = config.get_hash()

        model_dir = os.path.join(self.dir_, model_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        config_path = os.path.join(model_dir, 'config.yaml')
        if os.path.exists(config_path) and not force:
            raise FileExistsError(f'model already exists: {config_path} ')
        else:
            with open(config_path, 'w') as f:
                yaml.dump(config.to_dict(), f)
            print(f'Initialized {config_path}')

        return model_dir


class DatasetInitializer:
    def __init__(self):
        self.dir_ = get_config('config/paths.yaml').DATASETS

    def init_from_args(self, args):
        config = get_config('config/dataset.yaml')
        config.update_from_args(args, DatasetArgParser.arg_map)
        return self.write_config(args.kind, config, args.force)

    def init_from_kwargs(self, kind, force=False, **kwargs):
        config = get_config('config/dataset.yaml')
        for key, val in kwargs.items():
            config.set_field(DatasetArgParser.arg_map[key], val)
        return self.write_config(kind, config, force=force)

    def write_config(self, kind, config, force=False):
        dataset_id = config.get_hash()

        dataset_dir = os.path.join(self.dir_, kind, dataset_id)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        config_path = os.path.join(dataset_dir, 'config.yaml')
        if os.path.exists(config_path) and not force:
            raise FileExistsError(f'dataset already exists: {config_path} ')
        else:
            with open(config_path, 'w') as f:
                yaml.dump(config.to_dict(), f)
            print(f'Initialized {config_path}')

        return dataset_dir
