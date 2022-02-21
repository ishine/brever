import torch.nn as nn
import torch


class DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[1024, 1024],
                 dropout=0.2, batchnorm=False, batchnorm_momentum=0.1,
                 normalization='static'):
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
        if normalization == 'static':
            self.normalization = StaticNormalizer(input_size)
        elif normalization == 'cumulative':
            self.normalization = CumulativeNormalizer()
        else:
            raise ValueError('unrecognized normalization type, got '
                             f'{normalization}')

    def forward(self, x):
        x = self.normalization(x)
        x = x.transpose(1, 2)
        for operation in self.operations:
            x = operation(x)
        return x.transpose(1, 2)

    def enhance(self, x, dset, return_mask=False):
        mag, phase = dset.stft.analyze(x.unsqueeze(0))
        features = dset.feature_extractor((mag.squeeze(0), phase.squeeze(0)))
        features = dset.stack(features)
        mask = self.forward(features.unsqueeze(0))
        mask_extra = dset.mel_fb.extrapolate(mask)
        mag *= mask_extra
        x = dset.stft.synthesize((mag, phase))[..., :x.shape[-1]]
        if return_mask:
            return x.squeeze(0), mask.squeeze(0)
        else:
            x.squeeze(0)


class StaticNormalizer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        mean = torch.zeros((input_size, 1))
        std = torch.ones((input_size, 1))
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def set_statistics(self, mean, std):
        self.mean, self.std = mean, std

    def forward(self, x):
        return (x - self.mean)/self.std


class CumulativeNormalizer(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        batch_size, bins, frames = x.shape
        cum_sum = x.cumsum(-1)
        cum_pow_sum = x.pow(2).cumsum(-1)
        count = torch.arange(1, frames+1, device=x.device)
        count = count.reshape(1, 1, frames)
        cum_mean = cum_sum/count
        cum_var = cum_pow_sum/count - cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()
        return (x - cum_mean)/cum_std
