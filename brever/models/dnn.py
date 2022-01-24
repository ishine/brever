import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size, output_size, n_layers, dropout_toggle,
                 dropout_rate, dropout_input, batchnorm_toggle,
                 batchnorm_momentum, hidden_sizes):
        assert len(hidden_sizes) == n_layers
        super().__init__()
        self.operations = nn.ModuleList()
        if dropout_input:
            self.operations.append(nn.Dropout(dropout_rate))
        start_size = input_size
        for i in range(n_layers):
            end_size = hidden_sizes[i]
            self.operations.append(nn.Linear(start_size, end_size))
            if batchnorm_toggle:
                self.operations.append(
                    nn.BatchNorm1d(end_size, momentum=batchnorm_momentum))
            self.operations.append(nn.ReLU())
            if dropout_toggle:
                self.operations.append(nn.Dropout(dropout_rate))
            start_size = end_size
        self.operations.append(nn.Linear(start_size, output_size))
        self.operations.append(nn.Sigmoid())

    def forward(self, x):
        for operation in self.operations:
            x = operation(x)
        return x
