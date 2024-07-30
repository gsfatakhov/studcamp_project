import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class GNNTEP(nn.Module):
    def __init__(self, nnodes, window_size, ngnn, gsllayer, nhidden, alpha, k, out_channels, device,
                 constant_adj_matrix=None):
        super(GNNTEP, self).__init__()
        self.window_size = window_size
        self.nhidden = nhidden
        self.nnodes = nnodes
        self.device = device
        self.idx = torch.arange(self.nnodes).to(device)
        self.adj = [0 for i in range(ngnn)]
        self.h = [0 for i in range(ngnn)]
        self.skip = [0 for i in range(ngnn)]
        self.z = (torch.ones(nnodes, nnodes) - torch.eye(nnodes)).to(device)
        self.ngnn = ngnn

        self.graph_struct = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()

        if constant_adj_matrix is not None:
            self.constant_adj_matrix = torch.tensor(constant_adj_matrix, device=device, dtype=torch.float32)
        # else:
        #     self.constant_adj_matrix = torch.zeros((nnodes, nnodes), device=device, dtype=torch.float32)

        for i in range(self.ngnn):
            if gsllayer == 'relu':
                self.graph_struct.append(Graph_ReLu_W(nnodes, k, device, self.constant_adj_matrix))
            elif gsllayer == 'directed':
                self.graph_struct.append(Graph_Directed_A(nnodes, window_size, alpha, k, device))
            elif gsllayer == 'unidirected':
                self.graph_struct.append(Graph_Uni_Directed_A(nnodes, window_size, alpha, k, device))
            elif gsllayer == 'undirected':
                self.graph_struct.append(Graph_Undirected_A(nnodes, window_size, alpha, k, device))
            elif gsllayer == 'constant':
                self.graph_struct.append(ConstantGraph(self.constant_adj_matrix))

            else:
                print('Wrong name of graph structure learning layer!')
            self.conv1.append(GCNLayer(window_size, nhidden))
            self.bnorm1.append(nn.BatchNorm1d(nnodes))
            self.conv2.append(GCNLayer(nhidden, nhidden))
            self.bnorm2.append(nn.BatchNorm1d(nnodes))

        self.fc = nn.Linear(ngnn * nhidden, out_channels)

    def forward(self, X):

        X = X.to(self.device)

        for i in range(self.ngnn):
            self.adj[i] = self.graph_struct[i](self.idx)
            self.adj[i] = self.adj[i] * self.z
            self.h[i] = self.conv1[i](self.adj[i], X).relu()
            self.h[i] = self.bnorm1[i](self.h[i])
            self.skip[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.conv2[i](self.adj[i], self.h[i]).relu()
            self.h[i] = self.bnorm2[i](self.h[i])
            self.h[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.h[i] + self.skip[i]

        h = torch.cat(self.h, 1)
        output = self.fc(h)

        return output

    def get_adj(self):
        return self.adj
