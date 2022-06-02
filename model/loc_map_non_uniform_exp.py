import numpy as np
import scipy.io as scio
img_size=[1024,2048]
blk_size=[10,64]

y = np.linspace(0, img_size[0], num=blk_size[0] + 1)  # y==>H 序列生成器linspace(起点0，终点1024，个数11)
x = np.linspace(0, img_size[1], num=blk_size[1] + 1)  # x==>W 起点0，终点2048，个数65

a = []
for n in range(int((len(y) - 1)/2)):
    for m in range(2**(n+1)):
        a += [x[int((len(x)-1)/(2**(n+1)))*m], y[n], x[int((len(x)-1)/(2**(n+1)))*(m+1)], y[n + 1]]  # 从左到右从上到下扫描 [0 0 204.8 102.4 204.8 0 409.6 102.4...]

for n in range(int((len(y) - 1)/2),len(y)-1):
    for m in range(2**(blk_size[0]-n)):
        a += [x[int((len(x)-1)/(2**(blk_size[0]-n)))*m], y[n], x[int((len(x)-1)/(2**(blk_size[0]-n)))*(m+1)], y[n + 1]]  # 从左到右从上到下扫描 [0 0 204.8 102.4 204.8 0 409.6 102.4...]

a = np.array(a)
scio.savemat('loc_map.mat',{'a':a})
