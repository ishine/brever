import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout=0.0,
                 batchnorm=False, batchnorm_momentum=0.1):
        super().__init__()
        self.modules = nn.ModuleList()
        start_size = input_size
        for i in range(len(hidden_layers)):
            end_size = hidden_layers[i]
            self.modules.append(nn.Linear(start_size, end_size))
            if batchnorm:
                self.modules.append(
                    nn.BatchNorm1d(end_size, momentum=batchnorm_momentum)
                )
            self.modules.append(nn.ReLU())
            self.modules.append(nn.Dropout(dropout))
            start_size = end_size
        self.modules.append(nn.Linear(start_size, output_size))
        self.modules.append(nn.Sigmoid())

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
