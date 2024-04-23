import torch
from torch import nn
import numpy as np
import pandas as pd
#Padding incorrect?

class Model(torch.nn.Module):
    def __init__(self, structure: list) -> None:
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
    def __init__(self, inSize, outSize, kernelSize, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        mainlayers = [nn.Conv1d(inSize, outSize, kernelSize, stride=stride, padding=padding, dilation=dilation),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Conv1d(outSize, outSize, kernelSize, stride=stride, padding=padding, dilation=dilation),
                      nn.ReLU(),
                      nn.Dropout(dropout)]

        self.layers = torch.nn.Sequential(*mainlayers)

        self.downsample = None
        if inSize != outSize:
            self.downsample = nn.Conv1d(inSize, outSize, 1)
        self.__Initialise_Weights()

    def __Initialise_Weights(self):
        nn.init.kaiming_normal_(self.layers[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.layers[3].weight, mode='fan_out', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        residual = x
        out = self.layers(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        print(out.shape)
        print(residual.shape)
        out += residual
        out = self.relu(out)
        return out

class TCN(torch.nn.Module):
    def __init__(self, inputLength: int, outputLength: int, channels: list, kernelSize=2):
        super(TCN, self).__init__()
        layers = []
        depth = len(channels)

        for i in range(0, depth):
            dilation_size = 2 ** i
            inC = inputLength if i == 0 else channels[i-1]
            outC = channels[i]
            layers.append(TemporalBlock(inC, outC, kernelSize, stride=1, dilation=dilation_size, padding=(kernelSize-1) * dilation_size, dropout=0.2))
        
        layers.append(nn.Dropout(p=0.2))
        layers.append(nn.Linear(channels[-1], outputLength))
        self.network = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def calc(self, x):
        x = self.__tensorise(x)
        return self.network(x)

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


       
