import numpy as np
import random

a = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
b = [[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1]]

t = 0
random.seed(int(t))
random.shuffle(a)
random.seed(int(t))
random.shuffle(b)

print(a,b)

np.save('a.npy', a)
np.save('b.npy', b)
_a = np.load('a.npy')
_b = np.load('b.npy')

print(_a, _b)