import torch
from torch import nn
import numpy as np
import pandas as pd


class Model(torch.nn.Module):
    def __init__(self, modelType: str, structure: list) -> None:
        """
        modelType- basic, 
        structure- list of ints denoting size of each layer
        """

        super(Model, self).__init__()
            
        layers = []
            
        for i in range(len(structure) - 1):
            layers.append(nn.Linear(structure[i], structure[i + 1]))
            layers.append(nn.ReLU())
        layers.pop() #Remove excess RELU
            

        self.NN = torch.nn.Sequential(*layers)
            

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.__initialize_weights()

    def __initialize_weights(self):
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    nn.init.constant_(i.bias, 0)

    def calc(self, input):
        input = self.__tensorise(input)
        return self.NN(input)
    
    def train(self, yPred, yAct):
        yPred = self.__tensorise(yPred)
        yAct = self.__tensorise(yAct)

        self.optimizer.zero_grad()

        #print(yPred, yAct)
        loss = nn.functional.huber_loss(yPred, yAct)
        loss.backward()

        self.optimizer.step()
        return loss, [param.grad for param in self.parameters()]
    
    def __tensorise(self, x):
        '''
        Converts things to tensors (if they aren't already)
        '''
        if isinstance(x, pd.Series):
            x = x.tolist()
       
        if isinstance(x, list) or isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
    
        return x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        self.__initialise_weights()

    def __initialise_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')

    def calc(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class TCN(torch.nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def calc(self, x):
        return self.network(x)

    def train(self, train_loader, criterion, optimizer, num_epochs):
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

    def __tensorise(self, x):
        '''
        Converts things to tensors (if they aren't already)
        '''
        if isinstance(x, pd.Series):
            x = x.tolist()
       
        if isinstance(x, list) or isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
    
        return x


       
