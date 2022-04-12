from .convtasnet import ConvTasNet  # noqa
from .dnn import DNN  # noqa
from .dccrn import DCCRN  # noqa


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
