import numpy as np
import scipy.io as scio

img_size = [1336,2672]
blk_size = [10, 10]

y = np.linspace(0, img_size[0], num=blk_size[0] + 1)  # y==>H 序列生成器linspace(起点0，终点512，个数11)
#NN = [3,8,10,8,3] #y=5 X*3
#NN = [2,5,6,5,2] #y=5 X*2
#NN = [1,3,3,3,1]#y=5 X*1

#NN = [3, 9, 14, 17, 19, 19, 17, 14, 9, 3] #y=10,X*3
NN=[2,6,9,11,13,13,11,9,6,2] #y=10,X*2
#NN = [1,3,5,6,6,6,6,5,3,1]#y=10,X*1
#NN = [6, 18, 28, 34, 38, 38, 34, 28, 18, 6]


#NN=[ 3,9,14,19,23,26,28,29,28,26,23,19,14,9,3] #y=15,X*3
#NN=[2,6,10,13,15,17,19,19,19,17,15,13,10,6,2]#y=15,X*2 SUM:183
#NN=[1,3,5,6,8,9,9,10,9,9,8,6,5,3,1]#y=15,X*1
a = []
for n in range(len(y) - 1):
    x = np.linspace(0, img_size[1], num=NN[n] + 1)  # x==>W
    for m in range(len(x) - 1):
        a += [x[m], y[n], x[m + 1], y[n + 1]]  # 从左到右从上到下扫描

a = np.array(a)
scio.savemat('loc_map.mat', {'a': a})


