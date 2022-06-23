import torch

from brever.models import DNN, ConvTasNet, DCCRN


def test_dnn_forward():
    x = torch.randn(1, 64, 100)
    net = DNN()
    net(x)


def test_convtasnet_forward():
    x = torch.randn(1, 16000)
    net = ConvTasNet()
    net(x)


def test_dccrn_forward():
    x = torch.complex(
        torch.randn(1, 1, 512, 100),
        torch.randn(1, 1, 512, 100),
    )
    net = DCCRN()
    net(x)
