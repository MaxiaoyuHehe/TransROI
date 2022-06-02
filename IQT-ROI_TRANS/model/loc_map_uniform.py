import numpy as np
import scipy.io as scio

img_size = [1336, 2872]
blk_size = [10, 10]

y = np.linspace(0, img_size[0], num=blk_size[0] + 1)  # y==>H 序列生成器linspace(起点0，终点512，个数11)
x = np.linspace(0, img_size[1], num=blk_size[1] + 1)  # x==>W
a = []
for n in range(len(y) - 1):
    for m in range(len(x) - 1):
        a += [x[m], y[n], x[m + 1], y[n + 1]]  # 从左到右从上到下扫描

a = np.array(a)
scio.savemat('loc_map.mat', {'a': a})