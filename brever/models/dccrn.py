import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from .base import BreverBaseModel
from ..filters import STFT


class ComplexWrapper(nn.Module):
    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.module_real = module_cls(*args, **kwargs)
        self.module_imag = module_cls(*args, **kwargs)

    def forward(self, x):
        real = self.module_real(x.real) - self.module_imag(x.imag)
        imag = self.module_real(x.imag) + self.module_imag(x.real)
        return torch.complex(real, imag)


class SingleLayerComplexLSTM(ComplexWrapper):
    """
    Directly calling `ComplexWrapper` with `module_cls=nn.LSTM` leads to a
    wrong implementation of a multi-layer complex LSTM, since each layer needs
    to be wrapped. A single-layer complex LSTM is first defined here by
    sub-classing `ComplexWrapper`.

    Also `nn.LSTM` has multiple outputs so we need to rewrite `forward`.
    """
    def __init__(self, *args, **kwargs):
        if len(args) > 2 or 'num_layers' in kwargs:
            raise ValueError(f"{self.__class__.__name__} does not support "
                             "'num_layers' argument")
        super().__init__(nn.LSTM, *args, **kwargs)

    def forward(self, x):
        real_real, _ = self.module_real(x.real)
        imag_imag, _ = self.module_imag(x.imag)
        real_imag, _ = self.module_real(x.imag)
        imag_real, _ = self.module_imag(x.real)
        real = real_real - imag_imag
        imag = real_imag + imag_real
        return torch.complex(real, imag)


class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(SingleLayerComplexLSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False,
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ComplexBatchNorm2d(nn.Module):
    """
    Hats off to Ivan Nazarov for his implementation of complex BatchNorm:
    https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(2, 2, num_features))
            self.bias = torch.nn.Parameter(torch.empty(2, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean',
                                 torch.empty(2, num_features))
            self.register_buffer('running_var',
                                 torch.empty(2, 2, num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.copy_(torch.eye(2, 2).unsqueeze(-1))
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.copy_(torch.eye(2, 2).unsqueeze(-1))
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)"
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return complex_batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            exponential_average_factor,
            self.eps,
        )


def complex_batch_norm(input, running_mean, running_var, weight=None,
                       bias=None, training=True, momentum=0.1, eps=1e-05):
    assert ((running_mean is None and running_var is None)
            or (running_mean is not None and running_var is not None))
    assert ((weight is None and bias is None)
            or (weight is not None and bias is not None))

    # stack along the first axis
    x = torch.stack([input.real, input.imag], dim=0)

    # whiten and apply affine transformation
    z = whiten2x2(x, training=training, running_mean=running_mean,
                  running_cov=running_var, momentum=momentum, eps=eps)

    if weight is not None and bias is not None:
        shape = 1, x.shape[2], *([1] * (x.dim() - 3))
        weight = weight.reshape(2, 2, *shape)
        z = torch.stack([
            z[0] * weight[0, 0] + z[1] * weight[0, 1],
            z[0] * weight[1, 0] + z[1] * weight[1, 1],
        ], dim=0) + bias.reshape(2, *shape)

    return torch.complex(z[0], z[1])


def whiten2x2(tensor, training=True, running_mean=None, running_cov=None,
              momentum=0.1, eps=1e-5):
    # input must have shape (2, batch_size, channels, ...)
    assert tensor.dim() >= 3

    # the shape to reshape statistics to for broadcasting with real and
    # imaginary parts separately: (1, channels, 1, ..., 1)
    tail = 1, tensor.shape[2], *([1] * (tensor.dim() - 3))

    # the axes along which to average, i.e. all but 0 and 2
    axes = 1, *range(3, tensor.dim())

    # compute batch mean with shape (2, channels) and center the batch
    if training or running_mean is None:
        mean = tensor.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)
    else:
        mean = running_mean
    tensor = tensor - mean.reshape(2, *tail)

    # compute covariance matrix with shape (2, 2, channels)
    if training or running_cov is None:
        var = (tensor * tensor).mean(dim=axes) + eps
        cov_uu, cov_vv = var[0], var[1]
        cov_vu = cov_uv = (tensor[0] * tensor[1]).mean([a - 1 for a in axes])
        if running_cov is not None:
            cov = torch.stack([
                cov_uu.data, cov_uv.data,
                cov_vu.data, cov_vv.data,
            ], dim=0).reshape(2, 2, -1)
            running_cov += momentum * (cov - running_cov)
    else:
        cov_uu, cov_uv, cov_vu, cov_vv = running_cov.reshape(4, -1)

    # compute inverse square root R = [[p, q], [r, s]] of covariance matrix
    #
    # using Cholesky decomposition L.L^T=V seems to 'favour' the first
    # dimension i.e. the real part, so Trabelsi et al. (2018) used an explicit
    # inverse square root calculation as follows:
    #
    # for M = [[a, b], [c, d]] we have
    #     \sqrt{M} = \frac{1}{t} [[a+s, b], [c, d+s]]
    # where
    #     s = \sqrt{\det{M}} and t = \sqrt{\trace{M} + 2*s}
    # moreover it can be easily shown that
    #     \det{\sqrt{M}} = s
    # therefore using the formula of the inverse of a 2-by-2 matrix we have
    #     \inv{\sqrt{M}} = \frac{1}{ts} [[d+s, -b], [-c, a+s]]
    s = torch.sqrt(cov_uu*cov_vv - cov_uv*cov_vu)
    t = torch.sqrt(cov_uu + cov_vv + 2*s)
    denom = t*s
    p, q = (cov_vv+s)/denom, -cov_uv/denom
    r, s = -cov_vu/denom, (cov_uu+s)/denom

    # apply R = [[p, q], [r, s]] to input
    out = torch.stack([
        tensor[0] * p.reshape(tail) + tensor[1] * r.reshape(tail),
        tensor[0] * q.reshape(tail) + tensor[1] * s.reshape(tail),
    ], dim=0)
    return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super().__init__()
        self.conv = ComplexWrapper(
            module_cls=nn.Conv2d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = ComplexBatchNorm2d(out_channels)
        self.activation = ComplexWrapper(nn.PReLU)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding):
        super().__init__()
        self.conv = ComplexWrapper(
            module_cls=nn.ConvTranspose2d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.norm = ComplexBatchNorm2d(out_channels)
        self.activation = ComplexWrapper(nn.PReLU)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.lstm = ComplexLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.linear = ComplexWrapper(nn.Linear, hidden_size, input_size)

    def forward(self, x):
        x = self.lstm(x)
        x = self.linear(x)
        return x


class DCCRN(BreverBaseModel):
    """
    The original paper by Hu et al. (2020) advertises the network as fully
    complex, but the PReLU activations in the encoder/decoder are real.
    Moreover the dense layer after the LSTM is real in the implementation
    made available on GitHub.

    In this implementation, we use complex PReLU and dense layer modules, such
    that all operations are fully complex.

    Inspired from original code provided by authors Y. Hu et al. (2020):
    https://github.com/huyanxin/DeepComplexCRN
    """
    def __init__(
        self,
        criterion: str = 'SNR',
        stft_frame_length: int = 512,
        stft_hop_length: int = 128,
        stft_window: str = 'hann',
        channels: list[int] = [16, 32, 64, 128, 256, 256],
        kernel_size: tuple[int, int] = (5, 2),
        stride: tuple[int, int] = (2, 1),
        padding: tuple[int, int] = (2, 0),
        output_padding: tuple[int, int] = (1, 0),
        lstm_channels: int = 128,
        lstm_layers: int = 2,
    ):
        super().__init__(criterion)

        self.stft = STFT(
            frame_length=stft_frame_length,
            hop_length=stft_hop_length,
            window=stft_window,
        )

        stride_prod = math.prod(stride[0] for _ in channels)
        last_encoder_output_dim = stft_frame_length//2//stride_prod

        self.encoder = nn.ModuleList()
        for i in range(len(channels)):
            self.encoder.append(EncoderBlock(
                in_channels=1 if i == 0 else channels[i-1],
                out_channels=channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ))

        self.decoder = nn.ModuleList()
        for i in range(len(channels)-1, -1, -1):
            self.decoder.append(DecoderBlock(
                in_channels=channels[i]*2,
                out_channels=1 if i == 0 else channels[i-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ))

        self.lstm = LSTMBlock(
            input_size=channels[-1]*last_encoder_output_dim,
            hidden_size=lstm_channels,
            n_layers=lstm_layers,
        )

    def forward(self, input_):
        # input is (batch_size, channels, freq_bins, time)
        encoder_outputs = []
        x = input_
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_outputs.append(x)
        # permute to (batch_size, time, channels, freq_bins)
        x = x.permute(0, x.ndim-1, *range(1, x.ndim-1))
        # reshape to (batch_size, time, channels*freq_bins) for lstm, pass to
        # lstm and reshape back to original shape
        x = self.lstm(x.reshape(*x.shape[:2], -1)).reshape(*x.shape)
        # permute back to (batch_size, channels, freq_bins, time)
        x = x.permute(0, *range(2, x.ndim), 1)
        # pass to decoder
        for decoder_block, encoder_output in zip(
            self.decoder, reversed(encoder_outputs),
        ):
            x = torch.cat([x, encoder_output], axis=1)
            x = decoder_block(x)
        # apply complex mask
        output = x*input_
        output = F.pad(output, (0, 0, 0, 1))  # pad nyquist frequency
        output = self.stft.synthesize(output, input_type='complex')
        return output

    def pre_proc(self, data, target):
        data = self.stft.analyze(data.unsqueeze(0), return_type='complex')
        return data, target

    def enhance(self, x):
        # x has shape (channels, length)
        x = x.mean(axis=-2)  # (length,)
        x = x.view(1, 1, -1)  # (sources, channels, length)
        x = self.stft.analyze(x, return_type='complex')
        # (sources, channels, bins, length)
        x = x[..., :-1, :]  # remove nyquist frequency
        # current shape happens to work with shape needed for forward which is
        # (batch_size, channels (in the convolution sense), bins, length)
        x = self.forward(x)
        return x
