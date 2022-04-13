import torch
import torch.nn.functional as F

from brever.training import get_criterion


def _test_criterion(criterion):
    # seed
    torch.manual_seed(0)

    # params
    sources = 1
    length_1 = 3
    length_2 = 2  # < length_1
    lengths = [length_1, length_2]
    padding = length_1 - length_2

    # create 2 inputs of different length
    input_1 = torch.randn(1, sources, length_1)
    input_2 = torch.randn(1, sources, length_2)

    # pad shortest input
    input_2_padded = F.pad(input_2, (0, padding))

    # mimic neural net processing
    dummy_input_output_func = lambda x: x + torch.randn(*x.shape)  # noqa
    output_1 = dummy_input_output_func(input_1)
    output_2_padded = dummy_input_output_func(input_2_padded)
    output_2 = output_2_padded[..., :length_2]

    # create targets
    target_1 = torch.randn(1, sources, length_1)
    target_2 = torch.randn(1, sources, length_2)
    target_2_padded = F.pad(target_2, (0, padding))

    # 2 ways of calculating: either batch processing...
    batch_output = torch.cat([output_1, output_2_padded])
    batch_target = torch.cat([target_1, target_2_padded])
    criterion = get_criterion(criterion)

    # ...or one-by-one
    loss_a = criterion(batch_output, batch_target, lengths)
    loss_b = (
        criterion(output_1, target_1, [length_1]) +
        criterion(output_2, target_2, [length_2])
    )/2

    # both should give the same result
    assert loss_a == loss_b


def test_si_snr():
    _test_criterion('SISNR')


def test_snr():
    _test_criterion('SNR')


def test_mse():
    _test_criterion('MSE')
