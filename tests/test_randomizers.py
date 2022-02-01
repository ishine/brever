import pytest

from brever.mixture import (BaseRandGen, ChoiceRandGen, DistRandGen,
                            MultiChoiceRandGen, AngleRandomizer)


def test_errors():
    for rand in [
        BaseRandGen(),
        ChoiceRandGen(['foo', 'bar']),
        ChoiceRandGen(['foo', 'bar'], weights=[0.3, 0.7]),
        ChoiceRandGen(['foo', 'bar'], size=2, replace=True),
        ChoiceRandGen(['foo', 'bar'], size=2, replace=False),
        DistRandGen('uniform', [0.0, 1.0]),
        DistRandGen('logistic', [0.0, 4.3429448190325175]),
        MultiChoiceRandGen({'foo': [0, 1], 'bar': ['x', 'y', 'z']})
    ]:
        if isinstance(rand, MultiChoiceRandGen):
            def get():
                rand.get('foo')
        else:
            get = rand.get
        with pytest.raises(ValueError):
            get()
        rand.roll()
        get()
        with pytest.raises(ValueError):
            get()
        rand.roll()
        get()


def test_choice():
    rand = ChoiceRandGen(['foo', 'bar'], size=3, replace=False)
    with pytest.raises(ValueError):
        rand.roll()

    rand = ChoiceRandGen(['foo', 'bar'], size=10, weights=[0.0, 1.0])
    rand.roll()
    x = rand.get()
    assert 'foo' not in x

    rand = ChoiceRandGen(['foo', 'bar'], size=10, weights=[1.0, 1.0])
    rand.roll()
    x = rand.get()
    assert 'foo' in x and 'bar' in x

    rand = ChoiceRandGen(['foo', 'bar'], size=10)
    rand.roll()
    x = rand.get()
    assert 'foo' in x and 'bar' in x


def test_seed():
    rand1 = BaseRandGen(seed=0)
    rand2 = BaseRandGen(seed=0)
    rand1.roll()
    rand2.roll()
    assert rand1.get() == rand2.get()

    rand1 = BaseRandGen(seed=0)
    rand2 = BaseRandGen(seed=42)
    rand1.roll()
    rand2.roll()
    assert rand1.get() != rand2.get()

    rand = ChoiceRandGen(list(range(10)), size=2, seed=0)
    rand.roll()
    x1 = rand.get()
    rand.roll()
    x2 = rand.get()
    rand = ChoiceRandGen(list(range(10)), size=3, seed=0)
    rand.roll()
    y1 = rand.get()
    rand.roll()
    y2 = rand.get()
    assert x1 == y1[:2]
    assert x2 == y2[:2]

    rand1 = DistRandGen('uniform', [0.0, 1.0], seed=0)
    rand2 = DistRandGen('uniform', [0.0, 1.0], seed=0)
    rand1.roll()
    rand2.roll()
    assert rand1.get() == rand2.get()

    rand1 = DistRandGen('uniform', [0.0, 1.0], seed=0)
    rand2 = DistRandGen('uniform', [0.0, 1.0], seed=42)
    rand1.roll()
    rand2.roll()
    assert rand1.get() != rand2.get()

    rand1 = MultiChoiceRandGen(
        pool_dict={
            'foo': list(range(10)),
            'bar': list(range(42)),
        },
        seed=0,
    )
    rand2 = MultiChoiceRandGen(
        pool_dict={
            'foo': list(range(10)),
            'bar': list(range(42)),
        },
        seed=42,
    )
    rand1.roll()
    rand2.roll()
    assert rand1.get('bar') != rand2.get('bar')


def test_angles():
    target_angle = [-15, 15]
    noise_angle = [-30, 30]
    rand = AngleRandomizer(
        pool_dict={
            'surrey': list(range(-45, 45+5, 5)),
            'ash': list(range(-90, 90+10, 10)),
        },
        target_angle=target_angle,
        noise_angle=noise_angle,
        noise_num=3,
        parity='all',
        seed=0,
    )
    for i in range(10):
        rand.roll()
        t, ns = rand.get('surrey')
        assert target_angle[0] <= t <= target_angle[1]
        for n in ns:
            assert noise_angle[0] <= n <= noise_angle[1]

    rand = AngleRandomizer(
        pool_dict={
            'surrey': list(range(-45, 45+5, 5)),
            'ash': list(range(-90, 90+10, 10)),
        },
        noise_num=3,
        seed=0,
    )

    def exp():
        rand.roll()
        t, _ = rand.get('ash')
        return not (target_angle[0] <= t <= target_angle[1])

    assert any(exp() for i in range(10))

    def exp():
        rand.roll()
        _, ns = rand.get('surrey')
        for n in ns:
            if not (noise_angle[0] <= n <= noise_angle[1]):
                return True
        return False

    assert any(exp() for i in range(10))
