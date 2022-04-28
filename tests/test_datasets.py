from brever.data import DNNDataset


def test_dnn_dataset_segment_strategies():
    segment_strategies = [
        'drop',
        'pass',
        'pad',
        'overlap',
    ]
    lengths = [
        125,
        136,
        136,
        136,
    ]
    kwargs = {
        'segment_length': 0.25,
        'decimation': 1,
        'stft_frame_length': 512,
        'stft_hop_length': 256,
    }
    for segment_strategy, length in zip(segment_strategies, lengths):
        dataset = DNNDataset(
            'tests/test_dataset',
            segment_strategy=segment_strategy,
            **kwargs,
        )
        assert len(dataset) == length
        for x, y in dataset:
            break


def test_dnn_dataset_segment_length():
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
    kwargs = {
        'segment_strategy': 'pass',
        'decimation': 1,
        'stft_frame_length': 512,
        'stft_hop_length': 256,
    }
    for segment_length, length in zip(segment_lengths, lengths):
        dataset = DNNDataset(
            'tests/test_dataset',
            segment_length=segment_length,
            **kwargs,
        )
        assert len(dataset) == length
        for x, y in dataset:
            break
