from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, auc, save_pred
from torch.autograd import Variable
from models_cuda import GCN_single, GCN_hinge


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--start', type=int, default=0,
                    help='start in which turn')
parser.add_argument('--end', type=int, default=0,
                    help='end in which turn')
parser.add_argument('--device', type=int, default=0,
                    help='which GPU')


pair_num = 2
phase3 = False
single_loss_list = []

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    print(str(args.device))
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    

def train_single():
    output_list = []

    t = time.time()
    model_single.train()

    # optimize two by two
    for i in range(len(train_adj_list)):
        optimizer_single.zero_grad()

        adj = torch.Tensor(train_adj_list[i]).cuda()
        feature = torch.Tensor(train_feature_list[i]).cuda()
        output = model_single(feature, adj)
        output.squeeze_(0)
        output.squeeze_(0)
        loss_train = F.binary_cross_entropy_with_logits(output, torch.Tensor([train_label_list[i]]).cuda())

        loss_train.backward()
        optimizer_single.step()

    if phase3:
        for i in range(len(train_adj_list)):
            adj = torch.Tensor(train_adj_list[i]).cuda()
            feature = torch.Tensor(train_feature_list[i]).cuda()
            output = model_single(feature, adj)
            output_list.append(output)
        output_list = torch.Tensor(output_list).cuda()
        loss_train = F.binary_cross_entropy_with_logits(output_list, torch.Tensor(train_label_list).cuda())
        print("single_train_loss:", loss_train)

        if len(single_loss_list)>=5:
            single_loss_list.pop(0)
            single_loss_list.append(loss_train)
            if (max(single_loss_list)-min(single_loss_list))<0.05:
                print("convergence!")
                return True
            else:
                return False
        else:
            single_loss_list.append(loss_train)


def test_single(turn):
    output_list = []

    model_single.eval()
    for i in range(len(test_adj_list)):
        adj = torch.Tensor(test_adj_list[i]).cuda()
        feature = torch.Tensor(test_feature_list[i]).cuda()
        output = model_single(feature, adj)
        output_list.append(output)
    output_list = torch.Tensor(output_list)
    # loss_test = F.binary_cross_entropy_with_logits(output_list, torch.Tensor(test_label_list))
    # print("single_test_loss:", loss_test)

    labels = torch.Tensor(test_label_list)
    labels.unsqueeze_(0)
    output_list.unsqueeze_(0)
    test_accuracy = accuracy(output_list.t(), labels.t())
    print(test_accuracy)
    print()
    if phase3:
        save_pred(turn, output_list.t())
    return test_accuracy


train_adj_list, train_feature_list, train_label_list, test_adj_list, test_feature_list, test_label_list = load_data(0)
model_single = GCN_single(nfeat=4, nhid=args.hidden, nclass=2, dropout=args.dropout)
optimizer_single = optim.Adam(model_single.parameters(), lr=args.lr, weight_decay=args.weight_decay)
train_single()
test_accuracy = test_single(0)
print("test_done")


for turn in range(args.start, args.end):
    # Load data
    train_adj_list, train_feature_list, train_label_list, test_adj_list, test_feature_list, test_label_list = load_data(turn)

    model_single = GCN_single(nfeat=4,
                nhid=args.hidden,
                nclass=2,
                dropout=args.dropout)
    optimizer_single = optim.Adam(model_single.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        print("turn:", turn, ", epoch:", epoch)
        train_single()
        test_accuracy = test_single(turn)
        if (epoch>=10 and test_accuracy>=0.7):
        # if test_accuracy>=0.6:
            break
