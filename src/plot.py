
import numpy as np
import matplotlib.pyplot as plt 

x = []
single = np.load('data/0/res/pred0.npy')
    
print(len(single))
for i in range(len(single)):
    x.append(i)

# plt.plot(x,single,'r',label='single')
plt.scatter(x, single, alpha=0.6)

plt.title('The Lasers in Three Conditions')
plt.xlabel('row')
plt.ylabel('column')

# my_y_ticks = np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)

plt.legend()
plt.show()
