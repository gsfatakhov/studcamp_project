import torch
import torch.nn as nn
import torch.nn.functional as F


# Graph Convolutional Layer:
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        h = self.dense(X)
        norm = adj.sum(1) ** (-1 / 2)
        h = norm[None, :] * adj * norm[:, None] @ h
        return h


# A = ReLu(W)    
class Graph_ReLu_W(nn.Module):
    def __init__(self, num_nodes, k, device):
        super(Graph_Relu_W, self).__init__()
        self.num_nodes = num_nodes
        self.k = k

        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        adj = F.relu(self.A)

        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask

        return adj


# A for Directed graphs:
class Graph_Directed_A(nn.Module):

    def __init__(self, num_nodes, window_size, alpha, k, device):
        super(Graph_Directed_A, self).__init__()

        self.alpha = alpha
        self.k = k
        self.device = device

        self.e1 = nn.Embedding(num_nodes, window_size)
        self.e2 = nn.Embedding(num_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)
        self.l2 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha * self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha * self.l2(self.e2(idx)))
        adj = F.relu(torch.tanh(self.alpha * torch.mm(m1, m2.transpose(1, 0))))

        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask

        return adj


# A for Uni-directed graphs:
class Graph_Uni_Directed_A(nn.Module):

    def __init__(self, num_nodes, window_size, alpha, k, device):
        super(Graph_Directed_A, self).__init__()

        self.alpha = alpha
        self.k = k
        self.device = device

        self.e1 = nn.Embedding(num_nodes, window_size)
        self.e2 = nn.Embedding(num_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)
        self.l2 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha * self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha * self.l2(self.e2(idx)))
        adj = F.relu(torch.tanh(self.alpha * (torch.mm(m1, m2.transpose(1, 0)) - torch.mm(m2, m1.transpose(1, 0)))))

        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask

        return adj


# A for Undirected graphs:
class Graph_Undirected_A(nn.Module):

    def __init__(self, num_nodes, window_size, alpha, k, device):
        super(Graph_Directed_A, self).__init__()

        self.alpha = alpha
        self.k = k
        self.device = device

        self.e1 = nn.Embedding(num_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha * self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha * self.l1(self.e1(idx)))
        adj = F.relu(torch.tanh(self.alpha * torch.mm(m1, m2.transpose(1, 0))))

        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask

        return adj


class ConstantGraph(nn.Module):
    def __init__(self, constant_adj_matrix):
        super(ConstantGraph, self).__init__()
        self.constant_adj_matrix = constant_adj_matrix

    def forward(self, idx):
        return self.constant_adj_matrix
