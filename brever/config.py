import hashlib

import yaml


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


def get_config(path):
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    config = BreverConfig(config_dict)
    return config
