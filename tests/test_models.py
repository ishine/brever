import torch

from brever.models import DNN, ConvTasNet, DCCRN


def test_dnn_forward():
    x = torch.randn(1, 64, 100)
    net = DNN()
    net(x)


def test_dnn_enhance():
    x = torch.randn(2, 2000)
    net = DNN()
    net.enhance(x)


def test_convtasnet_forward():
    x = torch.randn(1, 16000)
    net = ConvTasNet()
    net(x)


def test_convtasnet_enhance():
    x = torch.randn(2, 2000)
    net = ConvTasNet()
    net.enhance(x)


def test_dccrn_forward():
    x = torch.complex(
        torch.randn(1, 1, 256, 100),
        torch.randn(1, 1, 256, 100),
    )
    net = DCCRN()
    net(x)


def test_dccrn_enhance():
    x = torch.randn(2, 2000)
    net = DCCRN()
    net.enhance(x)
