import logging

from .convtasnet import ConvTasNet  # noqa
from .dnn import DNN  # noqa
from .dccrn import DCCRN  # noqa


def initialize_model(config, dataset=None, train_split=None):
    logging.info('Initializing model')
    if config.ARCH == 'dnn':
        if dataset is None:
            raise ValueError("when config.ARCH is 'dnn', initialize_model "
                             "needs a dataset argument to set the number of "
                             "input and output nodes")
        model = DNN(
            input_size=dataset.n_features,
            output_size=dataset.n_labels,
            hidden_layers=config.MODEL.HIDDEN_LAYERS,
            dropout=config.MODEL.DROPOUT,
            batchnorm=config.MODEL.BATCH_NORM.TOGGLE,
            batchnorm_momentum=config.MODEL.BATCH_NORM.MOMENTUM,
            normalization=config.MODEL.NORMALIZATION.TYPE,
        )
        if train_split is not None:
            if config.MODEL.NORMALIZATION.TYPE == 'static':
                logging.info('Calculating training statistics')
                mean, std = train_split.dataset.get_statistics()
                model.normalization.set_statistics(mean, std)
    elif config.ARCH == 'convtasnet':
        model = ConvTasNet(
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
    else:
        raise ValueError(f'wrong model architecture, got {config.ARCH}')
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
