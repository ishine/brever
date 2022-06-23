from brever.args import DatasetArgParser, ModelArgParser
from brever.config import get_config


def test_dataset_args():
    parser = DatasetArgParser()

    assert len(parser._actions) == len(parser.arg_map) + 1

    arg_cmd = [
        '--duration', '0',
        '--seed', '0',
        '--speakers', 'foo', 'bar',
        '--noises', 'foo', 'bar',
        '--rooms', 'foo', 'bar',
        '--padding', '0',
        '--reflection-boundary', '0',
        '--uniform-tmr', '0',
        '--components', 'foo', 'bar',
        '--snr-dist-name', 'foo',
        '--snr-dist-args', '0', '0',
        '--target-angle', '0', '0',
        '--noise-num', '0', '0',
        '--noise-angle', '0', '0',
        '--ndr-dist-name', 'foo',
        '--ndr-dist-args', '0', '0',
        '--diffuse', '0',
        '--diffuse-color', 'foo',
        '--ltas-eq', '0',
        '--decay', '0',
        '--decay-color', 'foo',
        '--rt60-dist-name', 'foo',
        '--rt60-dist-args', '0', '0',
        '--delay-dist-name', 'foo',
        '--delay-dist-args', '0', '0',
        '--drr-dist-name', 'foo',
        '--drr-dist-args', '0', '0',
        '--rms-jitter-dist-name', 'foo',
        '--rms-jitter-dist-args', '0', '0',
        '--speech-files', '0', '0',
        '--noise-files', '0', '0',
        '--room-files', 'foo',
    ]
    args = parser.parse_args(arg_cmd)

    assert all(arg is not None for arg in args.__dict__.values())

    config = get_config('config/dataset.yaml')
    config.update_from_args(args, parser.arg_map)


def test_dnn_args():
    parser = ModelArgParser()

    arg_cmd = [
        # training args
        '--cuda', '0',
        '--early-stop', '0',
        '--convergence', '0',
        '--epochs', '0',
        '--learning-rate', '0',
        '--workers', '0',
        '--weight-decay', '0',
        '--train-path', 'foo',
        '--seed', '0',
        '--val-size', '0',
        '--criterion', 'foo',
        '--preload', '0',
        '--grad-clip', '0',
        '--optimizer', 'foo',
        '--segment-length', '0',
        '--batch-size', '0',
        '--batch-sampler', 'foo',
        '--sort-observations', '0',
        # model args
        'dnn',
        '--batch-norm', '0',
        '--dropout', '0',
        '--hidden-layers', '0', '0',
        '--norm-type', 'foo',
        '--group-norm', '0',
        '--features', 'foo', 'bar',
        '--decimation', '0',
        '--dct-coeff', '0',
        '--stacks', '0',
        '--scale-rms', '0',
    ]
    args = parser.parse_args(arg_cmd)

    assert all(arg is not None for arg in args.__dict__.values())

    config = get_config('config/models/dnn.yaml')
    config.update_from_args(args, parser.arg_map['dnn'])


def test_convtasnet_args():
    parser = ModelArgParser()

    arg_cmd = [
        # training args
        '--cuda', '0',
        '--early-stop', '0',
        '--convergence', '0',
        '--epochs', '0',
        '--learning-rate', '0',
        '--workers', '0',
        '--weight-decay', '0',
        '--train-path', 'foo',
        '--seed', '0',
        '--val-size', '0',
        '--criterion', 'foo',
        '--preload', '0',
        '--grad-clip', '0',
        '--optimizer', 'foo',
        '--segment-length', '0',
        '--batch-size', '0',
        '--batch-sampler', 'foo',
        '--sort-observations', '0',
        # model args
        'convtasnet',
        '--filters', '0',
        '--filter-length', '0',
        '--bottleneck-channels', '0',
        '--hidden-channels', '0',
        '--skip-channels', '0',
        '--kernel-size', '0',
        '--layers', '0',
        '--repeats', '0',
        '--sources', 'foo', 'bar',
    ]
    args = parser.parse_args(arg_cmd)

    assert all(arg is not None for arg in args.__dict__.values())

    config = get_config('config/models/convtasnet.yaml')
    config.update_from_args(args, parser.arg_map['convtasnet'])
