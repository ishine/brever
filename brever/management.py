import os
import hashlib

from .config import defaults


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


def get_unique_id(input_dict, n=8):
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
    n : int
        If provided, the first `n` characters from the ID are returns. Default
        is 8. If `None`, the whole ID is returned.

    Returns
    -------
    unique_id : str
        Unique ID of input dictionary.

    Notes
    -----
    Using `n=8`, the probability of obtaining 2 identical IDs after generating
    10000 random IDs is 1.2%. So `n=8` is the minimum recommended value.
    """
    if not input_dict:
        input_dict = {}
    input_dict = sorted_dict(input_dict)
    unique_str = ''.join([f'{hashlib.sha256(str(key).encode()).hexdigest()}'
                          f'{hashlib.sha256(str(val).encode()).hexdigest()}'
                          for key, val in input_dict.items()])
    unique_id = hashlib.sha256(unique_str.encode()).hexdigest()
    if n is None:
        n = len(unique_id)
    return unique_id[:n]


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


def find_model(models=None, configs=None, return_configs=False, **kwargs):
    """
    Find a model in the project model directory.

    Searches for models in the project model directory that match a set of
    parameter filters. The parameters must be provided as lists of possible
    values.

    Example: `find_model(layers=[1, 2])` returns the list of available models
    that use one or two hidden layers.

    Parameters
    ----------
    models: list of str, optional
        Pre-computed list of model paths to scan. If `None`, the whole
        project model directory is scanned. Default is `None`.
    configs: list of dict, optional
        Pre-computed list of model configuration dictionaries. If `None`,
        the YAML configuration file in each directory in `models` will be
        loaded. Default is `None`.
    return_configs: bool, optional
        If `True`, the list of model configuration dictionaries matching the
        filtering arguments is also returned. Default is `False`, which means
        only the list of matching paths is returned.
    **kwargs :
        Parameters to filter the list of available models. The available
        parameters are the keys of
        `~.modelmanagement.ModelFilterArgParser.arg_to_keys_map`,
        and must be set to lists of values of the appropriate type.

    Returns
    -------
    models : list of str
        List of model paths found in the project model directory that match
        the parameter filters.
    """
    if models is None:
        models = []
        models_dir = defaults().PATH.MODELS
        for model_id in os.listdir(models_dir):
            models.append(os.path.join(models_dir, model_id))

    if configs is None:
        configs = []
        for model in models:
            config_file = os.path.join(model, 'config_full.yaml')
            if not os.path.exists(config_file):
                config_file = os.path.join(model, 'config.yaml')
            config = read_yaml(config_file)
            configs.append(config)

    if not len(models) == len(configs):
        raise ValueError('models and configs must have same length, got '
                         f'{len(models)} and {len(configs)}')

    filtered_models = []
    filtered_configs = []
    for model, config in zip(models, configs):
        valid = True
        for key, value in kwargs.items():
            keys = ModelFilterArgParser.arg_to_keys_map[key]
            if value is not None and get_dict_field(config, keys) not in value:
                valid = False
                break
        if valid:
            filtered_models.append(model)
            filtered_configs.append(config)

    if return_configs:
        return filtered_models, filtered_configs
    else:
        return filtered_models


def find_dataset(kind=None, dsets=None, configs=None, return_configs=False,
                 **kwargs):
    """
    Find a dataset in the project dataset directory.

    Searches for datasets in the project dataset directory that match a set of
    parameter filters. Only one value per parameter can be provided
    simultaneously.

    Example: `find_dataset(rooms={surrey_room_a})` returns the list of datasets
    that only use the `surrey_room_a` room.

    Parameters
    ----------
    kind: {'train', 'val', 'test'}, optional
        Sub-directory to scan. Default is `None`, which means all
        sub-directories are scanned.
    dsets: list of str, optional
        Pre-computed list of dataset paths to scan. If `None`, the whole
        project dataset directory is scanned. Default is `None`.
    configs: list of dict, optional
        Pre-computed list of dataset configuration dictionaries. If `None`,
        the YAML configuration file in each directory in `dsets` will be
        loaded. Default is `None`.
    return_configs: bool, optional
        If `True`, the list of dataset configuration dictionaries matching the
        filtering arguments is also returned. Default is `False`, which means
        only the list of matching paths is returned.
    **kwargs :
        Parameters to filter the list of available datasets. The available
        parameters are the keys of
        `~.modelmanagement.DatasetInitArgParser.arg_to_keys_map`,
        and must be set as values of the appropriate type.

    Returns
    -------
    datasets : list of str
        List of dataset paths found in the project dataset directory that match
        the parameter filters.
    """
    if dsets is None:
        dsets = []
        directory = defaults().PATH.PROCESSED
        if kind is not None:
            directory = os.path.join(
                defaults().PATH.PROCESSED,
                kind,
            )
        for root, folder, files in os.walk(directory):
            if 'config.yaml' in files:
                dsets.append(root)

    if configs is None:
        configs = []
        for dset in dsets:
            config_file = os.path.join(dset, 'config_full.yaml')
            if not os.path.exists(config_file):
                config_file = os.path.join(dset, 'config.yaml')
            config = read_yaml(config_file)
            configs.append(config)

    if not len(dsets) == len(configs):
        raise ValueError('dsets and configs must have same length, got '
                         f'{len(dsets)} and {len(configs)}')

    filtered_dsets = []
    filtered_configs = []
    for dset, config in zip(dsets, configs):
        valid = True
        for key, value in kwargs.items():
            keys = DatasetInitArgParser.arg_to_keys_map[key]
            if value is not None:
                if get_dict_field(config, keys) != value:
                    valid = False
                    break
        if valid:
            filtered_dsets.append(dset.replace('\\', '/'))
            filtered_configs.append(config)

    if return_configs:
        return filtered_dsets, filtered_configs
    else:
        return filtered_dsets
