import numpy as np
import scipy.sparse as sp
import torch
from scipy.linalg import block_diag
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn import metrics

path="../data/"

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


def save_pred(turn, pred):
    np.save(path+'res/pred'+str(turn)+'.npy', pred)


def accuracy(output, labels, neg=0):
    pilot = -0.5
    # f = open("test.txt", "w")
    # for i in range(output.shape[0]):
    #     f.write(str(output[i]))
    #     f.write("\n")
    # f.close()
    # print(output.shape, labels.shape)
    preds = []
    for i in range(output.shape[0]):
        if output[i] > pilot:
            preds.append([1])
        else:
            preds.append([neg])
    preds = torch.Tensor(preds)
    return accuracy_score(labels, preds)


def auc(output, labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    fp0 = 0
    fp1 = 0
    fp2 = 0
    fp3 = 0
    fp4 = 0
    fp5 = 0
    fp6 = 0
    fp7 = 0
    fp8 = 0
    fp9 = 0

    fn0 = 0
    fn1 = 0
    fn2 = 0
    fn3 = 0
    fn4 = 0
    fn5 = 0
    fn6 = 0
    fn7 = 0
    fn8 = 0
    fn9 = 0

    preds = []
    for i in range(output.shape[0]):
        preds.append(output[i]-(-1)/2)
        if output[i]*labels[i]>=0:
            if labels[i]>0:
                tp = tp+1
            else:
                tn = tn+1
        else:
            if labels[i]>0:
                fn = fn+1
                if -output[i]<0.1:
                    fn0 = fn0+1
                elif -output[i]<0.2:
                    fn1 = fn1+1
                elif -output[i]<0.3:
                    fn2 = fn2+1
                elif -output[i]<0.4:
                    fn3 = fn3+1
                elif -output[i]<0.5:
                    fn4 = fn4+1
                elif -output[i]<0.6:
                    fn5 = fn5+1
                elif -output[i]<0.7:
                    fn6 = fn6+1
                elif -output[i]<0.8:
                    fn7 = fn7+1
                elif -output[i]<0.9:
                    fn8 = fn8+1
                elif -output[i]<1:
                    fn9 = fn9+1
            else:
                fp = fp+1
                if output[i]<0.1:
                    fp0 = fp0+1
                elif output[i]<0.2:
                    fp1 = fp1+1
                elif output[i]<0.3:
                    fp2 = fp2+1
                elif output[i]<0.4:
                    fp3 = fp3+1
                elif output[i]<0.5:
                    fp4 = fp4+1
                elif output[i]<0.6:
                    fp5 = fp5+1
                elif output[i]<0.7:
                    fp6 = fp6+1
                elif output[i]<0.8:
                    fp7 = fp7+1
                elif output[i]<0.9:
                    fp8 = fp8+1
                elif output[i]<1:
                    fp9 = fp9+1

    print("tp", tp, "tn", tn, "fp", fp, "fn", fn)

    print("fp0", format(fp0, '<5'), "\tfn0", fn0)
    print("fp1", format(fp1, '<5'), "\tfn1", fn1)
    print("fp2", format(fp2, '<5'), "\tfn2", fn2)
    print("fp3", format(fp3, '<5'), "\tfn3", fn3)
    print("fp4", format(fp4, '<5'), "\tfn4", fn4)
    print("fp5", format(fp5, '<5'), "\tfn5", fn5)
    print("fp6", format(fp6, '<5'), "\tfn6", fn6)
    print("fp7", format(fp7, '<5'), "\tfn7", fn7)
    print("fp8", format(fp8, '<5'), "\tfn8", fn8)
    print("fp9", format(fp9, '<5'), "\tfn9", fn9)

    preds = torch.Tensor(preds)
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    print("accuracy:", accuracy(output, labels, -1))
    return metrics.auc(fpr, tpr)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
