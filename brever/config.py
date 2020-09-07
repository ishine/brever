import yaml


class AttrDict:
    '''
    Immutable class initialized from a dictionary and implementing attribute
    access.
    '''
    def __init__(self, dict_):
        for key, value in dict_.items():
            if isinstance(value, dict):
                object.__setattr__(self, key, AttrDict(value))
            else:
                object.__setattr__(self, key, value)

    def __setattr__(self, attr, value):
        class_name = self.__class__.__name__
        raise AttributeError(f'{class_name} is an immutable class')

    def update(self, dict_):
        '''
        Attributes can only be modified via the update method, which takes as
        only argument a dictionary that is valid i.e. that respects the
        attribute hierarchy of the instance and the attribute types.
        '''
        if dict_ is None:
            return
        for key, value in dict_.items():
            if key not in self.__dict__:
                raise AttributeError(
                    f'{self.__class__.__name__} instance has no attribute {key}'
                )
            elif isinstance(self.__getattribute__(key), AttrDict):
                '''
                If the instance attribute is a nested AttrDict, then the value
                in the input dictionary must be a nested dictionary.
                '''
                if not isinstance(value, dict):
                    raise TypeError(
                        f'value with key {key} must have type dict, got {value.__class__.__name__}'
                    )
                self.__getattribute__(key).update(value)
            else:
                if self.__getattribute__(key).__class__ != value.__class__:
                    raise TypeError(
                        f'value with key {key} must have type {self.__getattribute__(key).__class__}, got {value.__class__.__name__}'
                    )
                else:
                    object.__setattr__(self, key, value)

    def to_dict(self):
        dict_ = {}
        for key, value in self.__dict__.items():
            if value.__class__ == self.__class__:
                dict_[key] = value.to_dict()
            else:
                dict_[key] = value
        return dict_


def defaults():
    with open('defaults.yaml') as f:
        dict_ = yaml.safe_load(f)
    return AttrDict(dict_)
