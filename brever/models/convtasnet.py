import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTasNet(nn.Module):
    def __init__(self, filters=512, filter_length=16, bottleneck_channels=128,
                 hidden_channels=512, skip_channels=128, kernel_size=3,
                 layers=8, repeats=3, sources=2):
        super().__init__()
        self.encoder = Encoder(filters, filter_length)
        self.decoder = Decoder(filters, filter_length)
        self.tcn = TCN(
            input_channels=filters,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            layers=layers,
            repeats=repeats,
            sources=sources,
        )

    def forward(self, x):
        length = x.shape[-1]
        x = self.encoder(x)
        masks = self.tcn(x)
        x = self.decoder(x, masks)
        x = x[:, :, :length]
        return x


class Encoder(nn.Module):
    def __init__(self, filters, filter_length, stride=None):
        super().__init__()
        if stride is None:
            stride = filter_length//2
        self.filter_length = filter_length
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=filters,
            kernel_size=filter_length,
            stride=stride,
            bias=False,
        )

    def pad(self, x):
        batch_size, length = x.shape
        # pad to obtain integer number of frames
        padding = (self.filter_length - length) % self.stride
        x = F.pad(x, (0, padding))  # pad left or right matters little here
        return x

    def forward(self, x):
        x = self.pad(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, filters, filter_length, stride=None):
        super().__init__()
        if stride is None:
            stride = filter_length//2
        self.filter_length = filter_length
        self.stride = stride
        self.trans_conv = nn.ConvTranspose1d(
            in_channels=filters,
            out_channels=1,
            kernel_size=filter_length,
            stride=stride,
            bias=False,
        )

    def forward(self, x, masks):
        batch_size, sources, channels, length = masks.shape
        x = x.unsqueeze(1)
        x = x*masks
        x = x.view(batch_size*sources, channels, length)
        x = self.trans_conv(x)
        x = x.view(batch_size, sources, -1)
        return x


class TCN(nn.Module):
    """
    Temporal convolutional network
    """
    def __init__(self, input_channels, bottleneck_channels, hidden_channels,
                 skip_channels, kernel_size, layers, repeats, sources):
        super().__init__()
        self.sources = sources
        self.layer_norm = cLN(input_channels)
        self.bottleneck_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
        )
        self.conv_blocks = nn.ModuleList()
        for b in range(repeats):
            for i in range(layers):
                dilation = 2**i
                self.conv_blocks.append(
                    Conv1DBlock(
                        input_channels=bottleneck_channels,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        dilation=dilation,
                        kernel_size=kernel_size,
                    )
                )
        self.prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=input_channels*sources,
            kernel_size=1,
        )

    def forward(self, x):
        batch_size, channels, length = x.shape
        x = self.layer_norm(x)
        x = self.bottleneck_conv(x)
        skip_sum = 0
        for conv_block in self.conv_blocks:
            x, skip = conv_block(x)
            skip_sum += skip
        x = self.prelu(skip_sum)
        x = self.output_conv(x)
        x = torch.sigmoid(x)
        return x.view(batch_size, self.sources, channels, length)


class cLN(nn.Module):
    """
    Cumulative layer normalization
    """
    def __init__(self, dim, eps=1e-10):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))
        self.eps = eps

    def forward(self, x):
        batch_size, channels, length = x.shape
        step_sum = x.sum(1)
        step_pow_sum = x.pow(2).sum(1)
        cum_sum = step_sum.cumsum(1)
        cum_pow_sum = step_pow_sum.cumsum(1)
        count = torch.arange(1, length+1, device=x.device)*channels
        count = count.reshape(1, -1)
        cum_mean = cum_sum/count
        cum_var = cum_pow_sum/count - cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        return (x - cum_mean)/cum_std*self.gain + self.bias


class Conv1DBlock(nn.Module):
    """
    1-D convolutional block
    """
    def __init__(self, input_channels, hidden_channels, skip_channels,
                 kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )
        self.d_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.res_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=input_channels,
            kernel_size=1,
        )
        self.skip_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=skip_channels,
            kernel_size=1,
        )
        self.norm_1 = cLN(hidden_channels)
        self.norm_2 = cLN(hidden_channels)
        self.prelu_1 = nn.PReLU()
        self.prelu_2 = nn.PReLU()

    def forward(self, x):
        # 1x1 convolution
        out = self.conv(x)
        out = self.prelu_1(out)
        out = self.norm_1(out)
        # pad to ensure residual has same size as input
        padding = (self.kernel_size - 1) * self.dilation
        out = F.pad(out, (padding, 0))  # pad left to ensure causality!
        # depthwise convolution
        out = self.d_conv(out)
        out = self.prelu_2(out)
        out = self.norm_2(out)
        # residual and skip connections
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        return x + res, skip
