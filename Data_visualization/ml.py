#!/usr/bin/env python
import os, csv
import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_lyrs = 1, do = .05, device = "cpu"):
        super(LSTM, self).__init__()

        self.ip_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_lyrs
        self.dropout = do
        self.device = device

        self.rnn = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_lyrs, dropout = do)
        self.fc1 = nn.Linear(in_features = hidden_dim, out_features = int(hidden_dim / 2))
        self.act1 = nn.ReLU(inplace = True)
        self.bn1 = nn.BatchNorm1d(num_features = int(hidden_dim / 2))

        self.estimator = nn.Linear(in_features = int(hidden_dim / 2), out_features = 3)
        
    
    def init_hiddenState(self, bs):
        return torch.ones(self.n_layers, bs, self.hidden_dim)

    def forward(self, input):
        input = input.unsqueeze(0) 
        bs = input.shape[1]
        hidden_state = self.init_hiddenState(bs).to(self.device)
        cell_state = hidden_state
        
        out, _ = self.rnn(input, (hidden_state, cell_state))

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.act1(self.bn1(self.fc1(out)))
        out = self.estimator(out)
        
        return out
    
    def predict(self, input):
        with torch.no_grad():
            predictions = self.forward(input)
        
        return predictions