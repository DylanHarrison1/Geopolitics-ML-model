import torch
from torch import nn
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tcn import TCN


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
            

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        self.__initialize_weights()

    def __initialize_weights(self):
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.kaiming_uniform_(i.weight)
                if i.bias is not None:
                    nn.init.constant_(i.bias, 0)

    def forward(self, input):
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
    """
    Blocks repeated throughout the TCN's structure
    """
    def __init__(self, inSize, outSize, kernelSize, stride, dilation, padding, dropout, layer, modelSize):
        super(TemporalBlock, self).__init__()
        self.layer = layer
        self.modelSize = modelSize

        #padding on 1st or 3rd
        mainlayers = [nn.ZeroPad1d((padding, 0)),
                      nn.Conv1d(inSize, outSize, kernelSize, stride=stride, padding=0, dilation=dilation),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.ZeroPad1d((padding, 0)),
                      nn.Conv1d(outSize, outSize, kernelSize, stride=stride, padding=0, dilation=dilation),
                      nn.ReLU(),
                      nn.Dropout(dropout)]
        #print(mainlayers)
        self.layers = torch.nn.Sequential(*mainlayers)

        self.downsample = None
        if inSize != outSize:
            self.downsample = nn.Conv1d(inSize, outSize, 1)

    def forward(self, x):
        #print(x.shape)
        residual = x
        out = self.layers(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        #print(out.shape)
        #print(residual.shape)
        #print("out " + str(out))
        #print("residual " + str(residual))

        out += residual

        #print("result " + str(out))
        if self.layer != self.modelSize - 1:
            relu = nn.ReLU()
            out = relu(out)
        return out
    

class TempConvNet(torch.nn.Module):
    def __init__(self, yearLength: int, indexNo: int, channels: list, kernelSize=3):
        super(TempConvNet, self).__init__()
        layers = []
        depth = len(channels)

        for i in range(0, depth):
            dilation = 2 ** (i)
            inC = indexNo if i == 0 else channels[i-1]
            outC = channels[i]
            layers.append(TemporalBlock(inC, outC, kernelSize, stride=1, dilation=dilation, padding=dilation * (kernelSize-1), dropout=0.20, layer=i, modelSize=depth ))
            
        
        self.network = nn.Sequential(*layers)
        #print(self.network)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.004999,)

        self.__initialiseWeights()
    
    def __initialiseWeights(self):
         for tempblock in self.modules():
             for i in tempblock.modules():
                 if isinstance(i, nn.Conv1d):
                    nn.init.normal_(i.weight)

    def forward(self, x):
        x = self.__tensorise(x)
        #print("inp " + str(x))
        #print("outp- " + str(self.network(x)[0]))
        #for layer in self.layers:
            #x = layer(x)
        return self.network(x)[0]

    def train(self, yPred, yAct):
        yPred = self.__tensorise(yPred)
        yAct = self.__tensorise(yAct)

        self.optimizer.zero_grad()
        
        #print(yPred.shape, yAct.shape)
        #print(yPred, yAct)
        #loss = nn.functional.huber_loss(yPred, yAct)
        loss = nn.functional.mse_loss(yPred, yAct)
        #print(loss)
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

