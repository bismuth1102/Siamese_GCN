import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import block_diag
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn import metrics

path="../../data/molecule/"

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(turn):
    """Load citation network dataset (cora only for now)"""
    print('Loading')

    # data = sio.loadmat("../../data/test/adj.mat")
    # adj = data['adj']
    # adj = torch.Tensor(adj)
    # print(adj.shape)

    train_adj_list = np.load(path+'train/'+str(turn)+'/adj.npy')
    train_feature_list = np.load(path+'train/'+str(turn)+'/feature.npy')
    train_label_list = np.load(path+'train/'+str(turn)+'/label.npy')

    test_adj_list = np.load(path+'test/'+str(turn)+'/adj.npy')
    test_feature_list = np.load(path+'test/'+str(turn)+'/feature.npy')
    test_label_list = np.load(path+'test/'+str(turn)+'/label.npy')


    return train_adj_list, train_feature_list, train_label_list, test_adj_list, test_feature_list, test_label_list


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def save_pred(sub_path, turn, phase, pred):
    np.save(path+str(sub_path)+str(turn)+'_'+str(phase)+'.npy', pred)


def accuracy(output, labels):
    preds = []
    m = nn.Sigmoid()
    for i in range(output.shape[0]):
        if m(torch.Tensor(output[i])) > 0.5:
            preds.append([1])
        else:
            preds.append([0])
    preds = torch.Tensor(preds)
    return accuracy_score(labels, preds)


def auc(output, labels):
    preds = []
    m = nn.Sigmoid()
    for i in range(output.shape[0]):
        preds.append(m(torch.Tensor(output[i])))
        
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    return metrics.auc(fpr, tpr)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
