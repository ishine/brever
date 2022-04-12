import torch
import torch.nn.functional as F

from brever.training import get_criterion


def _test_criterion(criterion):
    torch.manual_seed(0)
    sources = 1
    length_1 = 8
    length_2 = 4  # < length_1
    padding = length_1 - length_2
    x_1 = torch.randn(1, sources, length_1)
    x_2 = torch.randn(1, sources, length_2)
    x_2_padded = F.pad(x_2, (0, padding))
    y_1 = torch.randn(1, sources, length_1)
    y_2 = torch.randn(1, sources, length_2)
    y_2_padded = F.pad(y_2, (0, padding))
    batch_x = torch.cat([x_1, x_2_padded])
    batch_y = torch.cat([y_1, y_2_padded])
    lengths = [length_1, length_2]
    criterion = get_criterion(criterion)
    loss_a = criterion(batch_x, batch_y, lengths)
    loss_b = (
        criterion(x_1, y_1, [length_1]) +
        criterion(x_2, y_2, [length_2])
    )/2
    assert loss_a == loss_b


def test_si_snr():
    _test_criterion('SISNR')


def test_snr():
    _test_criterion('SNR')


def test_mse():
    _test_criterion('MSE')
