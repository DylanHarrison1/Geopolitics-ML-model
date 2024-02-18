import torch
from torch import nn
import numpy as np
import pandas as pd


class Model(torch.nn.Module):
    def __init__(self, inputDims) -> None:
        super(Model, self).__init__()

        layers = [
            nn.Linear(inputDims, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ]

        self.NN = torch.nn.Sequential(*layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

    def calc(self, input):
        input = self.__tensorise(input)
        return self.NN(input)
    
    def train(self, yPred, yAct):
        yPred, self.__tensorise(yPred)
        yAct, self.__tensorise(yAct)

        self.optimizer.zero_grad()
        loss = nn.functional.huber_loss(yPred, yAct)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def __tensorise(self, x):
        '''
        Converts things to tensors (if they aren't already)
        '''
        
        if isinstance(x, list) or isinstance(x, pd.Series):
            item = x.tensor(input)
        return x





       
