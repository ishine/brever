from brever.data import (BreverDataset, BreverBatchSampler, BreverDataLoader,
                         DNNDataset)


def test_audio_dataset():
    dataset = BreverDataset('tests/test_dataset')
    assert len(dataset) == 10

    batch_sampler = BreverBatchSampler(dataset, 3)
    dataloader = BreverDataLoader(dataset, batch_sampler=batch_sampler)
    assert len(dataloader) == 4
    for data, target in dataloader:
        pass

    batch_sampler = BreverBatchSampler(dataset, 3, drop_last=True)
    dataloader = BreverDataLoader(dataset, batch_sampler=batch_sampler)
    assert len(dataloader) == 3
    for data, target in dataloader:
        pass

    dataset = BreverDataset('tests/test_dataset', segment_length=4000)
    assert len(dataset) == 121

    batch_sampler = BreverBatchSampler(dataset, 3)
    dataloader = BreverDataLoader(dataset, batch_sampler=batch_sampler)
    assert len(dataloader) == 41
    for data, target in dataloader:
        pass

    batch_sampler = BreverBatchSampler(dataset, 3, drop_last=True)
    dataloader = BreverDataLoader(dataset, batch_sampler=batch_sampler)
    assert len(dataloader) == 40
    for data, target in dataloader:
        pass


def test_dnn_dataset():
    dataset = DNNDataset('tests/test_dataset', features={'logfbe'})
    assert len(dataset) == 10

    batch_sampler = BreverBatchSampler(dataset, 3)
    dataloader = BreverDataLoader(dataset, batch_sampler=batch_sampler)
    assert len(dataloader) == 4
    for data, target in dataloader:
        pass

    dataset = BreverDataset('tests/test_dataset', segment_length=4000)
    assert len(dataset) == 121

    batch_sampler = BreverBatchSampler(dataset, 3, drop_last=True)
    dataloader = BreverDataLoader(dataset, batch_sampler=batch_sampler)
    assert len(dataloader) == 40
    for data, target in dataloader:
        pass
