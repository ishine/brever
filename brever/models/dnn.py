import torch.nn as nn


class DNN(nn.Module):
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
