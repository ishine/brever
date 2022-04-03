import torch

from brever.models import DCCRN


def test_dccrn():
    net = DCCRN()
    x = torch.complex(
        torch.randn(16, 1, 512, 320),
        torch.randn(16, 1, 512, 320),
    )
    net(x)
