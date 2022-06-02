import os
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from model.model_main import IQARegression
import scipy.io as scio

from trainer import train_epoch, eval_epoch
from utils.util import RandCrop, RandHorizontalFlip, Normalize, ToTensor, RandShuffle, getConfig

# config file
config = getConfig()
# device setting
config.device = torch.device("cuda:%s" %config.GPU_ID if torch.cuda.is_available() else "cpu")

# data selection
if config.db_name == 'CVIQ':
    from data.data_CVIQ import IQADataset
elif config.db_name =='LIVE3D':
    from data.data_LIVE3D import IQADataset
elif config.db_name == 'LIVE':
    from data.data_LIVE import IQADataset
elif config.db_name == 'koniq':
    from data.data_koniq import IQADataset
elif config.db_name == 'MVAQD':
    from data.data_MVAQD import IQADataset



# data separation (8:2)
if not os.path.exists('train_list.mat'):
    train_scene_list, test_scene_list = RandShuffle(config.scenes, config.train_size) #返回两个数组，数组中是场景的序号
    scio.savemat('train_list.mat',{'train_list':train_scene_list})
    scio.savemat('test_list.mat', {'test_list': test_scene_list})
else:
    train_scene_list = scio.loadmat('train_list.mat')['train_list'].squeeze().tolist()
    test_scene_list = scio.loadmat('test_list.mat')['test_list'].squeeze().tolist()

# for reproducing results
# train_scene_list = [*range(160)]
# test_scene_list = [*range(160, 200)]
# train_scene_list = [*range(23)]
# test_scene_list = [*range(23, 29)]
print('number of train scenes: %d' % len(train_scene_list))
print('number of test scenes: %d' % len(test_scene_list))

# data load
train_dataset = IQADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    transform=transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), RandHorizontalFlip(), ToTensor()]),
    #transform=transforms.Compose([Normalize(0.5, 0.5), RandHorizontalFlip(), ToTensor()]),
    train_mode=True,
    scene_list=train_scene_list,
    train_size=config.train_size
)
test_dataset = IQADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    transform= transforms.Compose([Normalize(0.5, 0.5), ToTensor()]) if config.test_ensemble else transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), ToTensor()]),
    #transform= transforms.Compose([Normalize(0.5, 0.5), ToTensor()]) if config.test_ensemble else transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    train_mode=False,
    scene_list=test_scene_list,
    train_size=config.train_size
)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=False)


# create model
model_transformer = IQARegression(config).to(config.device) #这里只是进行了init，并没有进入forward


# loss function
criterion = torch.nn.MSELoss() #损失函数是MSE
optimizer = torch.optim.Adam(model_transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min) #用来改变学习率


# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)

# train & validation
test_srcc = np.zeros((config.n_epoch,1),dtype=np.float32) #numpy数组 100*1，用来存储每个epoch的test srcc的值
test_plcc = np.zeros((config.n_epoch,1),dtype=np.float32)

for epoch in range(0, config.n_epoch):
    loss, rho_s, rho_p = train_epoch(config, epoch, model_transformer, criterion, optimizer, scheduler, train_loader)

    if (epoch+1) % config.val_freq == 0: #这里val_freq=1,所以每个epoch都要做测试
        loss, rho_s, rho_p = eval_epoch(config, epoch, model_transformer, criterion, test_loader)
        test_srcc[epoch,0] = rho_s #保存每个epoch的测试srcc
        test_plcc[epoch, 0] = rho_p

scio.savemat('./results/plccs.mat',{'plcc':test_plcc})
scio.savemat('./results/srccs.mat',{'srcc':test_srcc})
print('END.....')


