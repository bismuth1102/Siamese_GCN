import torch
import scipy.io as sio

sio.savemat("adj.mat", {'adj': adj})
data = sio.loadmat("adj.mat")
adj = data['adj']
adj = torch.Tensor(adj)
print(adj[0][0])
