import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, frame_length=512, hop_length=256, window='hann',
                 hidden_layers=[1024, 1024], dropout=0.2, batchnorm=False,
                 batchnorm_momentum=0.1):
        self.stft = STFT(
            frame_length=frame_length,
            hop_length=hop_length,
            window=window,
            )
        self.ffnn = FFNN(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            dropout=dropout,
            batchnorm=batchnorm,
            batchnorm_momentum=batchnorm_momentum,
        )


class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[1024, 1024],
                 dropout=0.2, batchnorm=False, batchnorm_momentum=0.1):
        super().__init__()
        self.operations = nn.ModuleList()
        start_size = input_size
        for i in range(len(hidden_layers)):
            end_size = hidden_layers[i]
            self.operations.append(nn.Linear(start_size, end_size))
            if batchnorm:
                self.operations.append(
                    nn.BatchNorm1d(end_size, momentum=batchnorm_momentum)
                )
            self.operations.append(nn.ReLU())
            self.operations.append(nn.Dropout(dropout))
            start_size = end_size
        self.operations.append(nn.Linear(start_size, output_size))
        self.operations.append(nn.Sigmoid())
        self.transform = None

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = x.transpose(1, 2)
        for operation in self.operations:
            x = operation(x)
        return x.transpose(1, 2)


class STFT(nn.Module):
    def __init__(self, frame_length=512, hop_length=256, window='hann'):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length

        if isinstance(window, str):
            window = scipy.signal.get_window('hann', frame_length)**0.5
        if isinstance(window, np.ndarray):
            window = torch.from_numpy(window)
        self.window = window

        filters = torch.fft.fft(torch.eye(frame_length))
        filters = filters[:frame_length//2+1]
        filters[0, :] /= 2**0.5
        filters /= 0.5*frame_length/hop_length**0.5
        filters *= window
        filters = torch.cat([filters.real, filters.imag])

        filters = filters.unsqueeze(1).float()
        self.register_buffer("filters", filters)

    def analyze(self, x, return_type='realimag'):
        output = F.conv1d(x, self.filters, stride=self.hop_length)
        dim = self.frame_length//2 + 1
        real = output[:, :dim, :]
        imag = output[:, dim:, :]
        if return_type == 'realimag':
            return real, imag
        elif return_type == 'magphase':
            mag = (real.pow(2) + imag.pow(2)).pow(0.5)
            phase = torch.atan2(imag, real)
            return mag, phase
        else:
            raise ValueError("return_type must be 'realimag' or 'magphase', "
                             f", got '{return_type}'")

    def synthesize(self, x, input_type='realimag'):
        if input_type == 'realimag':
            real, imag = x
        elif input_type == 'magphase':
            mag, phase = x
            real = mag*torch.cos(phase)
            imag = mag*torch.sin(phase)
        else:
            raise ValueError("input_type must be 'realimag', 'complex' or "
                             f"'magphase', got '{input_type}'")
        x = torch.cat([real, imag], dim=1)
        return F.conv_transpose1d(x, self.filters, stride=self.hop_length)
