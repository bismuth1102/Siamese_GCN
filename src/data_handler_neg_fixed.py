from pysmiles import read_smiles
import networkx as nx
from one_hot import oneHot
import numpy as np
import linecache
import random
import scipy.io as io
import time

one_ten_of_pos = 760
pos_total = 7600
neg_total = 297941

class Data_handler():

	# compute & save A & X of smiles
	def res(self, smile, adj_list, feature_list):
		try:
			mol_with_H = read_smiles(smile, explicit_hydrogen=True)
		except:
			print(smile)
		A = nx.to_numpy_matrix(mol_with_H)
		X = oneHot(smile)
		adj_list.append(np.array(A))
		feature_list.append(np.array(X))

	# make the smile list randomly distributed
	def random(self, smile_list, label_list):
		t = time.time()
		# t = 0
		random.seed(int(t))
		random.shuffle(smile_list)
		random.seed(int(t))
		random.shuffle(label_list)


handler = Data_handler()
pos_list = []
neg_list = []


f = open("data/pos.txt", "r")
while True:
	smile = f.readline().strip()
	if not smile:
		break
	pos_list.append(smile)

random_nums = random.sample(range(1, neg_total), pos_total)
for line in random_nums:
	smile = linecache.getline("data/neg.txt", line).strip()
	neg_list.append(smile)

print(len(pos_list))
print(len(neg_list))


for turn in range(10):
	print("turn:",turn)

	train_smile_list = []
	train_adj_list = []
	train_feature_list = []
	train_label_list = []

	test_smile_list = []
	test_adj_list = []
	test_feature_list = []
	test_label_list = []

	handler = Data_handler()

	# test
	# in each test dataset, pos and neg are both 760
	test_smile_list.extend(pos_list[one_ten_of_pos*turn: one_ten_of_pos*(turn+1)])
	test_smile_list.extend(neg_list[one_ten_of_pos*turn: one_ten_of_pos*(turn+1)])
	test_label_list.extend([1]*one_ten_of_pos)
	test_label_list.extend([0]*one_ten_of_pos)


	# train
	# in each test dataset, pos and neg are both 760*9, but splited by test data
	train_smile_list.extend(pos_list[0: one_ten_of_pos*turn])
	train_smile_list.extend(pos_list[one_ten_of_pos*(turn+1): pos_total])
	train_smile_list.extend(neg_list[0: one_ten_of_pos*turn])
	train_smile_list.extend(neg_list[one_ten_of_pos*(turn+1): pos_total])
	train_label_list.extend([1]*one_ten_of_pos*9)
	train_label_list.extend([0]*one_ten_of_pos*9)

	handler.random(test_smile_list, test_label_list)
	handler.random(train_smile_list, train_label_list)

	for i in range(len(test_smile_list)):
		handler.res(test_smile_list[i], test_adj_list, test_feature_list)
	for i in range(len(train_smile_list)):
		handler.res(train_smile_list[i], train_adj_list, train_feature_list)


	print("test_smile_list", len(test_smile_list))
	print("test_adj_list", len(test_adj_list))
	print("test_feature_list", len(test_feature_list))
	print("test_label_list", len(test_label_list))

	print("train_smile_list", len(train_smile_list))
	print("train_adj_list", len(train_adj_list))
	print("train_feature_list", len(train_feature_list))
	print("train_label_list", len(train_label_list))


	np.save('data/test/' + str(turn) + '/adj.npy', test_adj_list)
	np.save('data/test/' + str(turn) + '/feature.npy', test_feature_list)
	np.save('data/test/' + str(turn) + '/label.npy', test_label_list)

	np.save('data/train/' + str(turn) + '/adj.npy', train_adj_list)
	np.save('data/train/' + str(turn) + '/feature.npy', train_feature_list)
	np.save('data/train/' + str(turn) + '/label.npy', train_label_list)
