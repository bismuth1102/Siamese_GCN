import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from sklearn import metrics
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--turn', type=int, default=0, help='which turn')
parser.add_argument('--phase', type=int, default=0, help='which phase')
parser.add_argument('--pilot', type=float, default=0, help='what is the pilot')

args = parser.parse_args()

path = '../../data/molecule/'
m = nn.Sigmoid()

def accuracy(output, labels):
    preds = []
    m = nn.Sigmoid()
    for i in range(len(output)):
        if m(torch.Tensor(output[i])) > 0.5:
            preds.append([1])
        else:
            preds.append([0])
    preds = torch.Tensor(preds)
    print("accuracy:", accuracy_score(labels, preds))


def auc(output, labels):
	preds = []
	for i in range(len(output)):
		preds.append(m(torch.Tensor(output[i])))
	fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
	print("auc:", metrics.auc(fpr, tpr))


# output = []
# labels = []
# output = np.load(path+'res_gcn/'+str(args.turn)+'_'+str(args.phase)+'.npy')
# labels = np.load(path+'test/'+str(args.turn)+'/label.npy')
# accuracy(output, labels)


output = []
labels = []
for i in range(0,10):
	temp_output = np.load(path+'res_gcn/'+str(i)+'_'+str(args.phase)+'.npy')
	output.extend(temp_output)
	temp_labels = np.load(path+'test/'+str(i)+'/label.npy')
	labels.extend(temp_labels)
accuracy(output, labels)
auc(output, labels)

