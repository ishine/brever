import pytest

from brever.config import BreverConfig


def test_hash():
    config_1 = {
        'foo': 0,
        'bar': 1,
    }
    config_2 = {
        'bar': 1,
        'foo': 0,
    }
    config_1 = BreverConfig(config_1)
    config_2 = BreverConfig(config_2)
    assert config_1.get_hash() == config_2.get_hash()

    config_1 = {
        'foo': 0,
    }
    config_2 = {
        'foo': 1,
    }
    config_1 = BreverConfig(config_1)
    config_2 = BreverConfig(config_2)
    assert config_1.get_hash() != config_2.get_hash()

    config_1 = {
        'foo': [0, 1],
    }
    config_2 = {
        'foo': [1, 0],
    }
    config_1 = BreverConfig(config_1)
    config_2 = BreverConfig(config_2)
    assert config_1.get_hash() != config_2.get_hash()

    config_1 = {
        'foo': {0, 1},
    }
    config_2 = {
        'foo': {1, 0},
    }
    config_1 = BreverConfig(config_1)
    config_2 = BreverConfig(config_2)
    assert config_1.get_hash() == config_2.get_hash()


def test_attribute_assignement():
    config = BreverConfig({'foo': 0})

    with pytest.raises(AttributeError):
        config.foo = 1

    with pytest.raises(TypeError):
        config.update_from_dict({'foo': 'bar'})

    config.update_from_dict({'foo': 1})
