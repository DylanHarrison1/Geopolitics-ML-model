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
            nn.Linear(20, 5)
        ]

        self.NN = torch.nn.Sequential(*layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.__initialize_weights()

    def __initialize_weights(self):
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
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
        return loss
    
    def __tensorise(self, x):
        '''
        Converts things to tensors (if they aren't already)
        '''
        if isinstance(x, pd.Series):
            x = x.tolist()
       
        if isinstance(x, list) or isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
    
        return x





       
