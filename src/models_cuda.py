import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np
from torch.autograd import Variable


class GCN_single(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, gc1_weight=None, gc2_weight=None, gc1_bias=None, gc2_bias=None):
        super(GCN_single, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, gc1_weight, gc1_bias).cuda()
        self.gc2 = GraphConvolution(nhid, 2, gc2_weight, gc2_bias).cuda()
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


class GCN_hinge(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, gc1_weight=None, gc2_weight=None, gc1_bias=None, gc2_bias=None):
        super(GCN_hinge, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, gc1_weight, gc1_bias).cuda()
        self.gc2 = GraphConvolution(nhid, 2, gc2_weight, gc2_bias).cuda()
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        x.unsqueeze_(0)
        pooling = nn.MaxPool2d((adj.shape[0],1))
        x = pooling(x)
        
        return x
