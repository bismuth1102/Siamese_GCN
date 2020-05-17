from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, auc, save_pred
from torch.autograd import Variable
from models_cheby_cuda import GCN_single, GCN_hinge


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=40,
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
                    help='use which GPU')

pair_num = 2
phase1 = False
phase3 = False
# single_loss_list = []

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)


def train_single(model, optimizer):
    output_list = []

    t = time.time()
    model.train()

    for i in range(len(train_adj_list)):
        optimizer.zero_grad()

        adj = torch.Tensor(train_adj_list[i]).cuda()
        feature = torch.Tensor(train_feature_list[i]).cuda()
        output = model(feature, adj)
        output.squeeze_(0)
        output.squeeze_(0)
        loss_train = F.binary_cross_entropy_with_logits(output, torch.Tensor([train_label_list[i]]).cuda())

        loss_train.backward()
        optimizer.step()

    # if phase3:
    #     for i in range(len(train_adj_list)):
    #         adj = torch.Tensor(train_adj_list[i]).cuda()
    #         feature = torch.Tensor(train_feature_list[i]).cuda()
    #         output = model(feature, adj)
    #         output_list.append(output)
    #     output_list = torch.Tensor(output_list).cuda()
    #     loss_train = F.binary_cross_entropy_with_logits(output_list, torch.Tensor(train_label_list).cuda())
    #     print("single_train_loss:", loss_train)

    #     if len(single_loss_list)>=5:
    #         single_loss_list.pop(0)
    #         single_loss_list.append(loss_train)
    #         diff = max(single_loss_list)-min(single_loss_list)
    #         print(diff)
    #         if (diff)<0.01:
    #             print("convergence!")
    #             return True
    #         else:
    #             return False
    #     else:
    #         single_loss_list.append(loss_train)


def test_single(model, turn):
    output_list = []

    model.eval()
    for i in range(len(test_adj_list)):
        adj = torch.Tensor(test_adj_list[i]).cuda()
        feature = torch.Tensor(test_feature_list[i]).cuda()
        output = model(feature, adj)
        output_list.append(output)

    labels = []
    output_list = torch.Tensor(output_list)
    labels = torch.Tensor(test_label_list)
    labels.unsqueeze_(0)
    output_list.unsqueeze_(0)

    print("accuracy:", accuracy(output_list.t(), labels.t()))
    print("auc:", auc(output_list.t(), labels.t()))

    # loss_test = F.binary_cross_entropy_with_logits(output_list, torch.Tensor(test_label_list))
    # print("single_test_loss:", loss_test)

    if phase1:
        save_pred("res_cheby/", turn, 1, output_list.t())
    elif phase3:
        save_pred("res_cheby/", turn, 3, output_list.t())


# get CosineSimilarity in pairs, save in a list
def get_cos_list(output_list, epoch):
    cos_list = []
    for i in range(len(output_list)):
        for j in range(i+1+pair_num*epoch, i+1+pair_num+pair_num*epoch):
            j = j%len(output_list)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_list.append([cos(output_list[i], output_list[j])])

    cos_list = torch.Tensor(cos_list)
    return cos_list


def train_hinge(epoch):
    output_list = []
    label = 0
    train_labels = [] #Cn_2 elements, ground truth for cos pairs in train

    t = time.time()
    model_hinge.train()

    # optimize two by two
    for i in range(len(train_adj_list)):
        for j in range(i+1+pair_num*epoch, i+1+pair_num+pair_num*epoch):
            j = j%len(train_adj_list)

            optimizer_hinge.zero_grad()

            adj1 = torch.Tensor(train_adj_list[i])
            feature1 = torch.Tensor(train_feature_list[i]).cuda()
            output1 = model_hinge(feature1, adj1)
            output1.squeeze_(0)

            adj2 = torch.Tensor(train_adj_list[j])
            feature2 = torch.Tensor(train_feature_list[j]).cuda()
            output2 = model_hinge(feature2, adj2)
            output2.squeeze_(0)

            if train_label_list[i]==train_label_list[j]:
                label = [1]
            else:
                label = [-1]
            train_labels.append(label)
            label = torch.Tensor(label)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            res_cos = cos(output1, output2)
            lossF = torch.nn.MarginRankingLoss(margin=0)
            loss_train = lossF(res_cos, torch.Tensor([0]).cuda(), torch.Tensor(label).cuda())

            loss_train.backward()
            optimizer_hinge.step()

    # # after one optimization, print the loss on all train samples
    # for i in range(len(train_adj_list)):
    #     adj = torch.Tensor(train_adj_list[i])
    #     # adj = train_adj_list[i]
    #     feature = torch.Tensor(train_feature_list[i])
    #     output = model_hinge(feature, adj)
    #     output.squeeze_(0)
    #     output_list.append(output)

    # cos_list = get_cos_list(output_list, epoch)
    # train_labels = torch.Tensor(train_labels)

    # lossF = torch.nn.MarginRankingLoss(margin=0)
    # # print("cos_list", cos_list.shape)
    # # print("train_labels", train_labels.shape)
    # loss_train = lossF(cos_list, torch.Tensor([0]), train_labels)

    # print(loss_train)


def test_hinge():
    output_list = []
    label = 0
    test_labels = [] #Cn_2 elements, ground truth for cos pairs in test

    model_hinge.eval()
    for i in range(len(test_adj_list)):
        adj = torch.Tensor(test_adj_list[i])
        feature = torch.Tensor(test_feature_list[i]).cuda()
        output = model_hinge(feature, adj)
        output.squeeze_(0)
        output_list.append(output)

    for i in range(len(test_adj_list)):
        for j in range(i+1, i+1+pair_num):
            j = j%len(test_adj_list)
            if test_label_list[i]==test_label_list[j]:
                test_labels.append([1])
            else:
                test_labels.append([-1])

    cos_list = get_cos_list(output_list, 0)
    test_labels = torch.Tensor(test_labels)

    # lossF = torch.nn.MarginRankingLoss(margin=0)
    # loss_test = lossF(cos_list, torch.Tensor([0]), test_labels)

    # print("test_loss:", loss_test)
    print("hinge_accuracy:", accuracy(cos_list, test_labels))
    print("hinge_auc:", auc(cos_list, test_labels))

    save_pred("res_cheby/", turn, 2, cos_list)


# print("test!")
# train_adj_list, train_feature_list, train_label_list, test_adj_list, test_feature_list, test_label_list = load_data(0)
# model_test = GCN_single(nfeat=4, nhid=args.hidden, nclass=2, dropout=args.dropout)
# optimizer_test = optim.Adam(model_test.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# train_single(model_test, optimizer_test)
# test_single(model_test, 0)
# print("test done!")
# print()


for turn in range(args.start, args.end):
    
    train_adj_list, train_feature_list, train_label_list, test_adj_list, test_feature_list, test_label_list = load_data(turn)

    phase1 = True
    model_single = GCN_single(nfeat=4,
                nhid=args.hidden,
                nclass=2,
                dropout=args.dropout)
    optimizer_single = optim.Adam(model_single.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total1 = time.time()
    for epoch in range(args.epochs):
        print("turn:", turn, ", epoch:", epoch)
        train_single(model_single, optimizer_single)
    print("phase1 time elapsed: {:.4f}s".format(time.time() - t_total1))
    test_single(model_single, turn)

    phase1 = False

    print("******************")

    model_hinge = GCN_hinge(nfeat=4,
                nhid=args.hidden,
                nclass=2,
                dropout=args.dropout)
    optimizer_hinge = optim.Adam(model_hinge.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total2 = time.time()
    for epoch in range(args.epochs):
        print("turn:", turn, ", epoch:", epoch)
        train_hinge(epoch)
    print("phase2 time elapsed: {:.4f}s".format(time.time() - t_total2))
    test_hinge()

    print("======================")

    phase3 = True
    model_single2 = GCN_single(nfeat=4,
                nhid=args.hidden,
                nclass=2,
                dropout=args.dropout)
    optimizer_single2 = optim.Adam(model_single2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total3 = time.time()
    for epoch in range(args.epochs):
        print("turn:", turn, ", epoch:", epoch)
        train_single(model_single2, optimizer_single2)
    print("phase3 time elapsed: {:.4f}s".format(time.time() - t_total3))
    test_single(model_single2, turn)

    phase3 = False
    print("##########################################")
