from .convtasnet import ConvTasNet  # noqa
from .dnn import DNN  # noqa
from .dccrn import DCCRN  # noqa


def initialize_model(config):
    if config.ARCH == 'dnn':
        model = DNN(
            criterion=config.TRAINING.CRITERION,
            fs=config.FS,
            features=config.MODEL.FEATURES,
            stacks=config.MODEL.STACKS,
            decimation=config.MODEL.DECIMATION,
            stft_frame_length=config.MODEL.STFT.FRAME_LENGTH,
            stft_hop_length=config.MODEL.STFT.HOP_LENGTH,
            stft_window=config.MODEL.STFT.WINDOW,
            mel_filters=config.MODEL.MEL_FILTERS,
            hidden_layers=config.MODEL.HIDDEN_LAYERS,
            dropout=config.MODEL.DROPOUT,
            normalization=config.MODEL.NORMALIZATION.TYPE,
        )
    elif config.ARCH == 'convtasnet':
        model = ConvTasNet(
            criterion=config.TRAINING.CRITERION,
            filters=config.MODEL.ENCODER.FILTERS,
            filter_length=config.MODEL.ENCODER.FILTER_LENGTH,
            bottleneck_channels=config.MODEL.TCN.BOTTLENECK_CHANNELS,
            hidden_channels=config.MODEL.TCN.HIDDEN_CHANNELS,
            skip_channels=config.MODEL.TCN.SKIP_CHANNELS,
            kernel_size=config.MODEL.TCN.KERNEL_SIZE,
            layers=config.MODEL.TCN.LAYERS,
            repeats=config.MODEL.TCN.REPEATS,
            sources=len(config.MODEL.SOURCES),
        )
    elif config.ARCH == 'dccrn':
        model = DCCRN(
            criterion=config.TRAINING.CRITERION,
            stft_frame_length=config.MODEL.STFT.FRAME_LENGTH,
            stft_hop_length=config.MODEL.STFT.HOP_LENGTH,
            stft_window=config.MODEL.STFT.WINDOW,
            channels=config.MODEL.CHANNELS,
            kernel_size=config.MODEL.KERNEL_SIZE,
            stride=config.MODEL.STRIDE,
            padding=config.MODEL.PADDING,
            output_padding=config.MODEL.OUTPUT_PADDING,
            lstm_channels=config.MODEL.LSTM_CHANNELS,
            lstm_layers=config.MODEL.LSTM_LAYERS,
        )
    else:
        raise ValueError(f'wrong model architecture, got {config.ARCH}')
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
