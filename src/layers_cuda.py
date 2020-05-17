import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, weight=None, bias=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weight==None:
            print("weight==None")
            self.weight_set = False
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        else:
            self.weight_set = True
            self.weight = weight

        if bias==None:
            print("bias==None")
            self.bias_set = False
            self.bias = Parameter(torch.cuda.FloatTensor(out_features))
        else:
            self.bias_set = True
            self.bias = bias

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.weight_set:
            self.weight.data.uniform_(-stdv, stdv)
        if not self.bias_set:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        # print(self.weight[0:4][0:4])
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
