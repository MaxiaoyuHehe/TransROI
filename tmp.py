import scipy.io as scio
import numpy as np
data = scio.loadmat('E:\\ImageDatabase\\LIVE3D\\dmos.mat')['dmos']
data[0,317]=50
print(np.max(data))
print(np.min(data))

print('sb')