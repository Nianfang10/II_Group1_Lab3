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
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers,dropout = 0.5, bidirectional=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=hidden_dim*2, out_features=output_size)
        self.classifier = nn.Softmax()

    def forward(self, X):
        out,hidden = self.gru(X)
        out = self.relu(out)
        out = out[:, -1, :]
        out = self.linear1(out)
        return out
    


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