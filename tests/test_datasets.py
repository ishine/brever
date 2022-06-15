from brever.data import BreverDataset


def test_segment_strategies():
    segment_strategies = [
        'drop',
        'pass',
        'pad',
        'overlap',
    ]
    lengths = [
        28,
        39,
        39,
        39,
    ]
    segment_length = 1.0
    for segment_strategy, length in zip(segment_strategies, lengths):
        dataset = BreverDataset(
            'tests/test_dataset',
            segment_strategy=segment_strategy,
            segment_length=segment_length,
        )
        assert len(dataset) == length
        for x, y in dataset:
            break


def test_segment_length():
    segment_lengths = [
        0.25,
        0.50,
        1.00,
        2.00,
    ]
    lengths = [
        136,
        72,
        39,
        23,
    ]
    for segment_length, length in zip(segment_lengths, lengths):
        dataset = BreverDataset(
            'tests/test_dataset',
            segment_length=segment_length,
        )
        assert len(dataset) == length
        for x, y in dataset:
            break
