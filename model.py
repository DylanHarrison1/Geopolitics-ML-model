import torch
from torch import nn
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Model, self).__init__()

        layers = [
            nn.Linear(inputDims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        ]

        self.NN = torch.nn.Sequential(*layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

    def calc(self, input):
        return self.NN(input)
    def train(self, yPred, yAct):
        self.optimizer.zero_grad()
        loss = nn.functional.huber_loss(yPred, yAct)
        loss.backward()
        self.optimizer.step()





       
