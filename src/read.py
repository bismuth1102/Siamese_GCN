import numpy as np
import argparse
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--turn', type=int, default=0, help='which turn')
parser.add_argument('--round', type=int, default=0, help='which round')
parser.add_argument('--pilot', type=float, default=0, help='which round')

args = parser.parse_args()

path = 'data/'

output = np.load(path+str(args.round)+'/res/pred'+str(args.turn)+'.npy')
labels = np.load(path+'test/'+str(args.turn)+'/label.npy')

def accuracy(output, labels, neg=0):
    preds = []
    for i in range(len(output)):
        if output[i] > args.pilot:
            preds.append(1)
        else:
            preds.append(neg)
    return accuracy_score(labels, preds)

print(len(output))
print(len(labels))

print(accuracy(output, labels))



