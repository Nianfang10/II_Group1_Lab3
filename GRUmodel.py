# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:25:49 2021

@author: ruswang
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        input_size = 4
        sequence_length = 71
        output_size = 52
        hidden_dim = 128
        n_layers = 2
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers,batch_first= True, dropout = 0.5, bidirectional=True)
        #self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=hidden_dim*2, out_features=output_size)
        #self.classifier = nn.Softmax()
        

    def forward(self, X):
       # print(X)
        #h0 = torch.zeros(self.n_layer, X.size(0),self.hidden_size)
        out,hidden = self.gru(X)        
        #out = self.relu(out)
        out = out[:, -1, :]
        out = self.linear1(out)
        return out
    
class UpdatingMean():
    def __init__(self) -> None:
        self.sum = 0
        self.n = 0

    def mean(self):
        return self.sum / self.n

    def add(self,loss):
        self.sum += loss
        self.n += 1


'''class BaseModel(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda=False):
        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        self.use_cuda = use_cuda
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=0.0,
                                 batch_first=True, )
        print(self,cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)
        
        
class GRUModel(BaseModel):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda):
        super(GRUModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda)
        
    def forward(self, x):

        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        if self.use_cuda:
            h0 = h0.cuda()
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput'''