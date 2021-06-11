import os
import hashlib
import argparse

import yaml

from brever.config import defaults


def sorted_dict(input_dict):
    """
    Sorts a dictionary by keys, and sorts the sets inside the dictionary.

    Recursively sorts a dictionary by keys (dictionaries in Python are actually
    ordered), and also converts the values of type `set` to sorted lists. This
    is done recursively, which means it works for nested dictionaries.

    This is useful to then obtain an unique hash ID from two equal
    dictionaries. Depending on the hash ID generation function, the ID can be
    different from one execution to another, or from two equal dictionaries
    defined in a different order, or containing sets defined in a different
    order (sets are randomly iterated through). As the order of elements in a
    set does not matter, converting them to sorted lists is not a problem and
    will make the hash ID consistent. Note this behavior is unwanted for lists,
    as we might want to obtain different IDs if the lists have the same
    elements but in a different order. So the lists inside the dictionary are
    not sorted.

    Parameters
    ----------
    input_dict : dict
        Input dictionary containing sets. Can be nested.

    Returns
    -------
    output_dict : dict
        Output dictionary with sets converted to sorted lists.
    """
    output_dict = {}
    for key, value in sorted(input_dict.items()):
        if isinstance(value, dict):
            output_dict[key] = sorted_dict(value)
        elif isinstance(value, set):
            output_dict[key] = sorted(value)
        else:
            output_dict[key] = value
    return output_dict


def get_unique_id(input_dict):
    """
    Unique ID from dictionary.

    Generates a unique ID (or at least attempts to) from a dictionary. A long
    string is first created by concatenating all the dictionary keys and values
    casted as hashed hexadecimal strings. The long string is then hashed into
    a final hexadecimal string to obtain the output ID.

    Parameters
    ----------
    input_dict : dict
        Input dictionary.

    Returns
    -------
    unique_id : str
        Unique ID of input dictionary.
    """
    if not input_dict:
        input_dict = {}
    input_dict = sorted_dict(input_dict)
    unique_str = ''.join([f'{hashlib.sha256(str(key).encode()).hexdigest()}'
                          f'{hashlib.sha256(str(val).encode()).hexdigest()}'
                          for key, val in input_dict.items()])
    unique_id = hashlib.sha256(unique_str.encode()).hexdigest()
    return unique_id


def flatten(input_dict, prefix=None):
    """
    Flatten a nested dictionary.

    Converts a nested dictionary to a flat, single-level dictionary. The keys
    of the output dictionary are obtained by concatenating the succesive
    original keys, separated by a dot (`'.'`).

    Example: the following dictionary `{'spam': {'foo': 0, 'bar': 1}}` is
    converted to `{'spam.foo': 0, 'spam.bar': 1}`.

    Parameters
    ----------
    input_dict : dict
        Input nested dictionary.
    prefix : str
        String to concatenate before each key. The string and the key are
        delimited with a dot. Used for the recursion. Default is `None`, which
        means no string is concatenated.

    Returns
    -------
    output_dict : dict
        Output flattened dictionary.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            for key, value in flatten(value, prefix=key).items():
                if prefix is None:
                    output_dict[key] = value
                else:
                    output_dict[f'{prefix}.{key}'] = value
        else:
            if prefix is None:
                output_dict[key] = value
            else:
                output_dict[f'{prefix}.{key}'] = value
    return output_dict


def unflatten(keys, values):
    """
    Unflatten multiple dictionaries.

    Recovers nested dictionaries from flat, single-level dictionaries with keys
    separated by dots (`'.'`).

    Parameters
    ----------
    keys : list of str
        List of keys. Each key should be a string consiting of atoms separated
        by dots.
    values : list of list
        List of list of values. Each element must be a list of values to use
        to recover a dictionary. This means each element must have the same
        length as `keys`.

    Returns
    -------
    output_dicts : list of dict
        List of nested dictionaries. Same length as `values`. All the
        dictionaries have the same keys, and the i-th dictionary uses the i-th
        list of values in `values`.
    """
    output_dicts = []
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
        output_dicts.append(config)
    return output_dicts


def set_dict_field(input_dict, key_list, value):
    """
    Set a nested dictionary value.

    Sets a field in a nested dictionary given a list of keys to use to reach
    the field. If while navigating the dictionary, the current key does not
    exist, then a new inner dictionary is created.

    Parameters
    ----------
    input_dict : dict
        Input dictionary.
    key_list : list of str
        List of keys to use to navigate in the nested dictionary and reach the
        value.
    value : any type
        Value to assign.
    """
    dict_ = input_dict
    for key in key_list:
        if key not in dict_.keys():
            dict_[key] = {}
        if key == key_list[-1]:
            dict_[key] = value
        else:
            dict_ = dict_[key]


def get_dict_field(input_dict, key_list, default=None):
    """
    Get a nested dictionary value.

    Gets a field in a nested dictionary given a list of keys to use to reach
    the field.

    Parameters
    ----------
    input_dict : dict
        Input dictionary.
    key_list : list of str
        List of keys to use to navigate in the nested dictionary and reach the
        value.
    default : any type, optional
        The default value to return if, while navigating the dictionary, the
        current key in `key_list` does not exist. Default is `None`.

    Returns
    -------
    value : any type
        Value found in `input_dict` using the path `key_list`.
    """
    try:
        dict_ = input_dict
        for key in key_list:
            if key == key_list[-1]:
                return dict_[key]
            else:
                dict_ = dict_[key]
    except KeyError:
        return default


def set_config_field(input_config_dict, argument_tag, value):
    """
    Set a configuration dictionary value.

    Sets a field in a configuration dictionary given an argument. The list of
    available arguments are available in
    `~.modelmanagement.ModelFilterArgParser.arg_to_keys_map`. A configuration
    dictionary can be obtained from e.g. `~.config.AttrDict.to_dict`. If while
    navigating the dictionary, the current key does not exist, then a new inner
    dictionary is created.

    Parameters
    ----------
    input_config_dict : dict
        Input configuration dictionary.
    argument_tag : str
        Argument tag. The list of available arguments and their corresponding
        path in the configuration dictionary are available in
        `~.modelmanagement.ModelFilterArgParser.arg_to_keys_map`.
    value : any type
        Value to assign.
    """
    set_dict_field(input_config_dict,
                   ModelFilterArgParser.arg_to_keys_map[argument_tag], value)


def get_config_field(input_config_dict, argument_tag, default=None):
    """
    Get a configuration dictionary value.

    Gets a field in a configuration dictionary given a list of keys to use to
    reach the field. A configuration dictionary can be obtained from e.g.
    `~.config.AttrDict.to_dict`.

    Parameters
    ----------
    input_config_dict : dict
        Input configuration dictionary.
    argument_tag : str
        Argument tag. The list of available arguments and their corresponding
        path in the configuration dictionary are available in
        `~.modelmanagement.ModelFilterArgParser.arg_to_keys_map`.
    default : any type, optional
        The default value to return if the value associated with
        `argument_tag` does not exist. Default is `None`.

    Returns
    -------
    value : any type
        Value found in `input_config_dict` associated with `argument_tag`.
    """
    return get_dict_field(input_config_dict,
                          ModelFilterArgParser.arg_to_keys_map[argument_tag],
                          default)


def find_model(**kwargs):
    """
    Find a model in the project model directory.

    Searches for models in the project model directory that match a set of
    parameter filters. The parameters must be provided as lists of possible
    values.

    Example: `find_model(layers=[1, 2])` returns the list of available models
    that use one or two hidden layers.

    Parameters
    ----------
    **kwargs :
        Parameters to filter the list of available models. The available
        parameters are the keys of
        `~.modelmanagement.ModelFilterArgParser.arg_to_keys_map`,
        and must be set to lists of values of the appropriate type.

    Returns
    -------
    models : list of str
        List of model IDs found in the project model directory that match the
        parameter filters. These are model IDs, i.e. these are not the paths
        to the model directories. To obtain the paths to the model directories,
        these should be joined with the project model directory path.
    """
    models_dir = defaults().PATH.MODELS
    models = []
    for model_id in os.listdir(models_dir):
        config_file = os.path.join(models_dir, model_id, 'config_full.yaml')
        if not os.path.exists(config_file):
            config_file = os.path.join(models_dir, model_id, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        valid = True
        for key, value in kwargs.items():
            keys = ModelFilterArgParser.arg_to_keys_map[key]
            if value is not None and get_dict_field(config, keys) not in value:
                valid = False
                break
        if valid:
            models.append(model_id)
    return models


class ExtendableArgParser(argparse.ArgumentParser):
    """
    Argument parser with extra group of arguments.

    A subclass of `argparse.ArgumentParser` that can be extended to have a
    second group of arguments. The arguments are denoted as either base
    arguments or extra arguments.

    Note this could be improved by extending to the general case of any number
    of group of arguments.

    Also note the desired behavior is not achievable with the built-in
    argument groups and sub-parsers of `argparse`, because the parsed
    arguments are then returned together in a single namespace, without the
    group information they belong to.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_dests = []
        self._extra_dests = []

    def add_base_argument(self, *args, **kwargs):
        """
        Add a base argument. Takes the same arguments and keyword arguments as
        the `add_argument` method of `argparse.ArgumentParser`.
        """
        _storeAction = super().add_argument(*args, **kwargs)
        self._base_dests.append(_storeAction.dest)

    def add_argument(self, *args, **kwargs):
        """
        Add an extra argument. Overwrites the `add_argument` method of
        `argparse.ArgumentParser`. Takes the same arguments and keyword
        arguments as the `add_argument` method of `argparse.ArgumentParser`.
        """
        _storeAction = super().add_argument(*args, **kwargs)
        if args == ('-h', '--help'):
            return
        self._extra_dests.append(_storeAction.dest)

    def parse_args(self, *args, **kwargs):
        """
        Parse arguments. Overwrites the `parse_args` method of
        `argparse.ArgumentParser`. Takes the same arguments and keyword
        arguments as the `parse_args` method of `argparse.ArgumentParser`.

        Returns
        -------
        base_args : argparse.Namespace
            Namespace containing base arguments.
        extra_args : argparse.Namespace
            Namespace containing extra arguments.
        """
        args = super().parse_args(*args, **kwargs)
        base_args = argparse.Namespace()
        extra_args = argparse.Namespace()
        for dest in self._base_dests:
            setattr(base_args, dest, getattr(args, dest))
        for dest in self._extra_dests:
            setattr(extra_args, dest, getattr(args, dest))
        return base_args, extra_args


def arg_set_type(input_str):
    """
    A convenience function that casts a string of words separated by spaces to
    a set of strings. This is useful for an argument parser that should return
    an argument as a set.

    Parameters
    ----------
    input_str : str
        Input string.

    Returns
    -------
    output_set: set of str
        Set of strings whose elements are obtained by splitting `input_str`
        using blank spaces (`' '`) as the delimiter. Empty strings are excluded
        from the output.
    """
    output_set = set(input_str.split(' '))
    if '' in output_set:
        output_set.remove('')
    return output_set


def arg_filelims_type(input_str):
    """
    A convenience function that checks if the input limits are valid. Limits
    should come as a string of two floats separated by a space. The two floats
    must be different, ascending and between 0.0 and 1.0. Used to parse dataset
    file limits arguments.

    Parameters
    ----------
    input_str : input_str
        Input string of two floats separated by a space.

    Returns
    -------
    output_list: list of float
        List of two floats.
    """
    output_list = input_str.split(' ')
    while '' in output_list:
        output_list.remove('')
    assert len(output_list) == 2
    output_list = [float(item) for item in output_list]
    assert output_list[0] < output_list[1]
    assert 0 <= output_list[0] <= 1
    assert 0 <= output_list[1] <= 1
    return output_list


def arg_list_type(input_str, post_caster):
    """
    A convenience function that casts the input string of words separated by
    spaces into a list of strings. Used to parse arguments as lists.

    Parameters
    ----------
    input_str : input_str
        Input string consisting of words separated by spaces.
    post_caster: callable
        And additional function to use on the individual substrings.

    Returns
    -------
    output_list: list of str
        List of strings.
    """
    output_list = input_str.split(' ')
    while '' in output_list:
        output_list.remove('')
    output_list = [post_caster(item) for item in output_list]
    return output_list


class ModelFilterArgParser(ExtendableArgParser):
    """
    Model filter argument parser.

    Subclass of `~.modelmanagement.ExtendableArgParser` that is ready to take
    as arguments the available model parameters in
    `~.modelmanagement.ModelFilterArgParser.arg_to_keys_map`. This can be
    extended to accept additional arguments for eventual further processing.

    Typical usage example is to feed the base arguments to
    `~.modelmanagement.find_model` and then use the extra arguments for
    further processing of the found models, e.g. training or testing them.
    """

    arg_to_keys_map = {
        'layers': ['MODEL', 'NLAYERS'],
        'stacks': ['POST', 'STACK'],
        'batchnorm': ['MODEL', 'BATCHNORM', 'ON'],
        'dropout': ['MODEL', 'DROPOUT', 'ON'],
        'dropout_rate': ['MODEL', 'DROPOUT', 'RATE'],
        'dropout_input': ['MODEL', 'DROPOUT', 'INPUT'],
        'batchsize': ['MODEL', 'BATCHSIZE'],
        'features': ['POST', 'FEATURES'],
        'labels': ['POST', 'LABELS'],
        'train_path': ['POST', 'PATH', 'TRAIN'],
        'val_path': ['POST', 'PATH', 'VAL'],
        'test_path': ['POST', 'PATH', 'TEST'],
        'dct': ['POST', 'DCT', 'ON'],
        'n_dct': ['POST', 'DCT', 'NCOEFF'],
        'cuda': ['MODEL', 'CUDA'],
        'uni_norm_features': ['POST', 'STANDARDIZATION', 'UNIFORMFEATURES'],
        'file_based_norm': ['POST', 'STANDARDIZATION', 'FILEBASED'],
        'epochs': ['MODEL', 'EPOCHS'],
        'earlystop': ['MODEL', 'EARLYSTOP', 'ON'],
        'progress': ['MODEL', 'PROGRESS', 'ON'],
        'weight_decay': ['MODEL', 'WEIGHTDECAY'],
        'decimation': ['POST', 'DECIMATION'],
        'learning_rate': ['MODEL', 'LEARNINGRATE'],
        'seed': ['MODEL', 'SEED'],
        'prestack': ['POST', 'PRESTACK'],
        'load': ['POST', 'LOAD'],
        'hidden_sizes': ['MODEL', 'HIDDENSIZES'],
    }

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
            '--dropout-rate',
            type=float,
            nargs='+',
            help='dropout rate (between 0.0 and 1.0)',
        )
        self.add_base_argument(
            '--dropout-input',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='dropout input layer toggle',
        )
        self.add_base_argument(
            '--batchsize',
            type=int,
            nargs='+',
            help='batchsize',
        )
        self.add_base_argument(
            '--features',
            type=arg_set_type,
            nargs='+',
            help='feature set',
        )
        self.add_base_argument(
            '--labels',
            type=arg_set_type,
            nargs='+',
            help='label set',
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
            '--test-path',
            type=lambda x: x.replace('\\', '/'),
            nargs='+',
            help='testing dataset path',
        )
        self.add_base_argument(
            '--dct',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='dct toggle',
        )
        self.add_base_argument(
            '--n-dct',
            type=int,
            nargs='+',
            help='number of dct coefficients',
        )
        self.add_base_argument(
            '--cuda',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='cuda toggle',
        )
        self.add_base_argument(
            '--uni-norm-features',
            type=arg_set_type,
            nargs='+',
            help='features to uniformly normalize',
        )
        self.add_base_argument(
            '--file-based-norm',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='file-based normalization toggle',
        )
        self.add_base_argument(
            '--epochs',
            type=int,
            nargs='+',
            help='maximum number of epochs',
        )
        self.add_base_argument(
            '--earlystop',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='early stopping toggle',
        )
        self.add_base_argument(
            '--progress',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='progress tracker toggle',
        )
        self.add_base_argument(
            '--weight-decay',
            type=float,
            nargs='+',
            help='weight decay',
        )
        self.add_base_argument(
            '--decimation',
            type=int,
            nargs='+',
            help='decimation factor',
        )
        self.add_base_argument(
            '--learning-rate',
            type=float,
            nargs='+',
            help='learning rate',
        )
        self.add_base_argument(
            '--seed',
            type=int,
            nargs='+',
            help='seed',
        )
        self.add_base_argument(
            '--prestack',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='pre-stack before training loop',
        )
        self.add_base_argument(
            '--load',
            type=lambda x: bool(int(x)),
            nargs='+',
            help='load dataset into memory',
        )
        self.add_base_argument(
            '--hidden-sizes',
            type=lambda x: arg_list_type(x, int),
            nargs='+',
            help='list of sizes of hidden layers',
        )


class DatasetInitArgParser(ExtendableArgParser):
    """
    Dataset initialization argument parser.

    Subclass of `~.modelmanagement.ExtendableArgParser` that is ready to
    take as arguments parameters for dataset initialization. This can be
    extended to accept additional arguments for eventual further processing.
    """

    arg_to_keys_map = {
        'decay': ['PRE', 'MIXTURES', 'DECAY', 'ON'],
        'decay_color': ['PRE', 'MIXTURES', 'DECAY', 'COLOR'],
        'diffuse': ['PRE', 'MIXTURES', 'DIFFUSE', 'ON'],
        'diffuse_color': ['PRE', 'MIXTURES', 'DIFFUSE', 'COLOR'],
        'rt60_min': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'RT60', 'MIN'],
        'rt60_max': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'RT60', 'MAX'],
        'delay_min': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'DELAY', 'MIN'],
        'delay_max': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'DELAY', 'MAX'],
        'rooms': ['PRE', 'MIXTURES', 'RANDOM', 'ROOMS'],
        'noise_types': ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES', 'TYPES'],
        'random_rms': ['PRE', 'MIXTURES', 'RANDOM', 'RMSDB', 'ON'],
        'rms_min': ['PRE', 'MIXTURES', 'RANDOM', 'RMSDB', 'MIN'],
        'rms_max': ['PRE', 'MIXTURES', 'RANDOM', 'RMSDB', 'MAX'],
        'scale_rms': ['PRE', 'MIXTURES', 'SCALERMS'],
        'n_sources_min': ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES', 'NUMBER', 'MIN'],
        'n_sources_max': ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES', 'NUMBER', 'MAX'],
        'filelims_noise': ['PRE', 'MIXTURES', 'FILELIMITS', 'NOISE'],
        'filelims_target': ['PRE', 'MIXTURES', 'FILELIMITS', 'TARGET'],
        'features': ['PRE', 'FEATURES'],
        'seed': ['PRE', 'SEED', 'ON'],
        'seed_value': ['PRE', 'SEED', 'VALUE'],
        'snr_dist_name': ['PRE', 'MIXTURES', 'RANDOM', 'TARGET', 'SNR', 'DISTNAME'],
        'snr_dist_args': ['PRE', 'MIXTURES', 'RANDOM', 'TARGET', 'SNR', 'DISTARGS'],
        'drr_dist_name': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'DRR', 'DISTNAME'],
        'drr_dist_args': ['PRE', 'MIXTURES', 'RANDOM', 'DECAY', 'DRR', 'DISTARGS'],
        'uniform_tmr': ['PRE', 'MIXTURES', 'RANDOM', 'UNIFORMTMR'],
        'target_angle_min': ['PRE', 'MIXTURES', 'RANDOM', 'TARGET', 'ANGLE', 'MIN'],
        'target_angle_max': ['PRE', 'MIXTURES', 'RANDOM', 'TARGET', 'ANGLE', 'MAX'],
        'target_datasets': ['PRE', 'MIXTURES', 'RANDOM', 'TARGET', 'DATASETS'],
        'noise_angle_min': ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES', 'ANGLE', 'MIN'],
        'noise_angle_max': ['PRE', 'MIXTURES', 'RANDOM', 'SOURCES', 'ANGLE', 'MAX'],
    }

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
            type=arg_set_type,
            help='list of rooms',
        )
        self.add_base_argument(
            '--noise-types',
            type=arg_set_type,
            help='noise types',
        )
        self.add_base_argument(
            '--random-rms',
            type=lambda x: bool(int(x)),
            help='rms randomization toggle',
        )
        self.add_base_argument(
            '--rms-min',
            type=int,
            help='minimum rms offset',
        )
        self.add_base_argument(
            '--rms-max',
            type=int,
            help='maximum rms offset',
        )
        self.add_base_argument(
            '--scale-rms',
            type=lambda x: bool(int(x)),
            help='rms scaling toggle',
        )
        self.add_base_argument(
            '--n-sources-min',
            type=int,
            help='minimum number of noise sources',
        )
        self.add_base_argument(
            '--n-sources-max',
            type=int,
            help='maximum number of noise sources',
        )
        self.add_base_argument(
            '--filelims-noise',
            type=arg_filelims_type,
            help='noise recordings file limits',
        )
        self.add_base_argument(
            '--filelims-target',
            type=arg_filelims_type,
            help='speech recordings file limits',
        )
        self.add_base_argument(
            '--features',
            type=arg_set_type,
            help='features to generate',
        )
        self.add_base_argument(
            '--seed',
            type=lambda x: bool(int(x)),
            help='seed toggle',
        )
        self.add_base_argument(
            '--seed-value',
            type=int,
            help='seed value',
        )
        self.add_base_argument(
            '--snr-dist-name',
            type=str,
            help='target snr distribution name',
        )
        self.add_base_argument(
            '--snr-dist-args',
            type=float,
            nargs='+',
            help='target snr distribution arguments',
        )
        self.add_base_argument(
            '--drr-dist-name',
            type=str,
            help='decay drr distribution name',
        )
        self.add_base_argument(
            '--drr-dist-args',
            type=float,
            nargs='+',
            help='decay drr distribution arguments',
        )
        self.add_base_argument(
            '--uniform-tmr',
            type=lambda x: bool(int(x)),
            help='impose a uniform tmr distribution',
        )
        self.add_base_argument(
            '--target-angle-min',
            type=float,
            help='minimum target direction angle',
        )
        self.add_base_argument(
            '--target-angle-max',
            type=float,
            help='maximum target direction angle',
        )
        self.add_base_argument(
            '--target-datasets',
            type=arg_set_type,
            help='target speech datasets',
        )
        self.add_base_argument(
            '--noise-angle-min',
            type=float,
            help='minimum noise direction angle',
        )
        self.add_base_argument(
            '--noise-angle-max',
            type=float,
            help='maximum noise direction angle',
        )
