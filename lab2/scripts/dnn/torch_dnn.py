import torch
import torch.nn as nn


class TorchDNN(nn.Module):
    """Create a DNN to extract posteriors that can be used for HMM decoding
    Parameters:
        input_dim (int): Input features dimension
        output_dim (int): Number of classes
        num_layers (int): Number of hidden layers
        batch_norm (bool): Whether to use BatchNorm1d after each hidden layer
        hidden_dim (int): Number of neurons in each hidden layer
        dropout_p (float): Dropout probability for regularization
    """
    def __init__(
        self, input_dim, output_dim, num_layers=2, batch_norm=True, hidden_dim=[256], dropout_p=0.2
    ):
        super(TorchDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dnn = nn.Sequential()
        for i in range(num_layers-1):
            if i == 0:
                self.dnn.add_module('fc{}'.format(i+1), nn.Linear(self.input_dim, hidden_dim[i]))
            else:
                self.dnn.add_module('fc{}'.format(i+1), nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            if batch_norm:
                self.dnn.add_module('bn{}'.format(i+1), nn.BatchNorm1d(hidden_dim[i]))
            self.dnn.add_module('relu{}'.format(i+1), nn.ReLU())
            self.dnn.add_module('dropout{}'.format(i+1), nn.Dropout(dropout_p))

        self.dnn.add_module('fc{}'.format(i+2), nn.Linear(hidden_dim[i], self.output_dim))

    def forward(self, x):
        return self.dnn(x)
