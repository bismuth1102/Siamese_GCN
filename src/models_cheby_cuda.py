import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_cuda import GraphConvolution
import numpy as np
import dgl
from dgl.nn.pytorch.conv import ChebConv
import scipy.sparse as ss

class GCN_single(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_single, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid).cuda()
        self.gc2 = GraphConvolution(nhid, 2).cuda()
        self.gc3 = nn.Linear(2, 1).cuda()
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        x.unsqueeze_(0)
        pooling = nn.MaxPool2d((adj.shape[0],1))
        x = pooling(x)
        x = self.gc3(x)
        
        return x

    def check(self):
        self.gc1.check()
        self.gc2.check()


class GCN_hinge(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_hinge, self).__init__()

        self.gc1 = ChebConv(nfeat, nhid, 3).cuda()
        self.gc2 = GraphConvolution(nhid, 2).cuda()
        self.dropout = dropout

    def forward(self, x, adj):
        adj_ss = ss.coo_matrix((adj), shape=(adj.shape[0],adj.shape[1]))
        adj = torch.Tensor(adj).cuda()
        g = dgl.DGLGraph()
        g.from_scipy_sparse_matrix(adj_ss)
        
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        x.unsqueeze_(0)
        pooling = nn.MaxPool2d((adj.shape[0],1))
        x = pooling(x)
        
        return x
