import torch
import torch.nn as nn


class ComplexWrapper(nn.Module):
    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.module_real = module_cls(*args, **kwargs)
        self.module_imag = module_cls(*args, **kwargs)

    def forward(self, x):
        real = self.module_real(x.real) - self.module_imag(x.imag)
        imag = self.module_real(x.imag) + self.module_imag(x.real)
        return torch.complex(real, imag)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, *args, which='naive', **kwargs):
        super().__init__()
        if which == 'naive':
            self.norm = ComplexWrapper(nn.BatchNorm2d, *args, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.norm(x)


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
        self.activation = nn.PReLU()

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
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, layers):
        super().__init__()
        self.lstms = nn.ModuleList()
        for i in range(layers):
            lstm = ComplexWrapper(
                module_cls=nn.LSTM,
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False,
            )
            self.lstms.append(lstm)

        def forward(self, x):
            for lstm in self.lstms:
                x = lstm(x)
            return x


class DCCRN(nn.Module):
    def __init__(
        self,
        channels=[1, 16, 32, 64, 128, 256, 256],
        kernel_size=(5, 2),
        stride=(2, 1),
        padding=(2, 0),
        output_padding=(1, 0),
        lstm_layers=2,
        lstm_channels=128
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        for i in range(len(channels)-1):
            encoder_block = EncoderBlock(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.encoder.append(encoder_block)

        self.decoder = nn.ModuleList()
        for i in range(len(channels)-1, 0, -1):
            decoder_block = DecoderBlock(
                in_channels=channels[i]*2,
                out_channels=channels[i-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
            self.decoder.append(decoder_block)

        self.lstm = LSTMBlock(
            input_size=channels[-1],
            hidden_size=lstm_channels,
            layers=lstm_layers,
        )

    def forward(self, x):
        for encoder_block in self.encoder:
            x = encoder_block(x)
        x = self.lstm(x)
        for decoder_block in self.decoder:
            x = decoder_block(x)
