# used code from https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/

# import linear algebra and data manipulation libraries
import numpy as np
import pandas as pd

# import pytorch
import torch
from torch import nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):

    #Our batch shape for input x is (1, 28, 28)

    def __init__(self, hidden_size = 64, output_size = 10):
        '''
        Init method
        INPUT:
            hidden_size - size of the hidden fully-connnected layer
            output_size - size of the output
        '''
        super(SimpleCNN, self).__init__()

        #Input channels = 3, output channels = 18
        self.conv1 = nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #3528 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(18 * 14 * 14, hidden_size)

        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
        Forward pass of the model.
        INPUT:
            x - input data
        '''

        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 14 * 14)

        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)
