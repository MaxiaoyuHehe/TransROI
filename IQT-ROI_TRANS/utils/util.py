import torch
import numpy as np
from option.config import Config
def getConfig():
    config = Config({
        # device
        "GPU_ID": "0",
        "num_workers": 0,

        "n_enc_seq": 82,  # patch的个数
        "n_dec_seq": 82,  #
        "n_layer": 2,  # number of encoder/decoder layers
        "d_hidn": 256,  # input channel (C) of encoder / decoder (input: C x N)
        "i_pad": 0,
        "d_ff": 1024,  # feed forward hidden layer dimension
        "d_MLP_head": 256,  # hidden layer of final MLP
         #"n_head": 6,# number of head (in multi-head attention)
        "n_head": 4,
        "d_head": 256,  # input channel (C) of each head (input: C x N) -> same as d_hidn
        "dropout": 0.1,  # dropout ratio of transformer
        "emb_dropout": 0.1,  # dropout ratio of input embedding
        "layer_norm_epsilon": 1e-12,
        "n_output": 1,  # dimension of final prediction
        #"crop_size": (1024,2048),  # input image crop size for LIVE3D & CVIQ &MVAQD
        # "crop_size": (1336,2872)
        "crop_size": (1336,2672),  # input image crop size for LIVE3D
        #"crop_size": (512,683),  # input image crop size for koniq

        # data CVIQ
      # "db_name": "CVIQ",  # database name [ PIPAL | LIVE | CSIQ | TID2013 ]
      # "db_path": "E:\\ImageDatabase\\CVIQ\\CVIQ_database\\CVIQ",  # root of dataset
      # "snap_path": "./weights/CVIQ",  # path for saving weights
      # "txt_file_name": "./IQA_list/CVIQ.txt",  # image list file (.txt)
      # "train_size": 0.8,
      # "scenes": list(range(16)),

        # data LIVE3D test_ensenmble=false
        #"db_name": "LIVE3D",
        #"db_path": "E:\\ImageDatabase\\LIVE3D\\Images\\Images\\All_Images",  # root of dataset
        #"snap_path": "./weights/LIVE3D",  # path for saving weights
        #"txt_file_name": "./IQA_list/LIVE3D.txt",  # image list file (.txt)
        #"train_size": 0.8,
        #"scenes": list(range(15)),

        # data LIVE
        #"db_name": "LIVE",
        #"db_path": "E:\\ImageDatabase\\live",  # root of dataset
        #"snap_path": "./weights/LIVE",  # path for saving weights
        #"txt_file_name": "./IQA_list/LIVE_IQA.txt",  # image list file (.txt)
        #"train_size": 0.8,
        #"scenes": list(range(29)),

        # data koniq
        #"db_name": "koniq",  # database name [ PIPAL | LIVE | CSIQ | TID2013 ]
        #"db_path": "E:\\ImageDatabase\\1024x768",  # root of dataset
        #"snap_path": "./weights/koniq",  # path for saving weights
        #"txt_file_name": "./IQA_list/koniq-10k.txt",  # image list file (.txt)
        #"train_size": 0.8,
        #"scenes": list(range(10073)),

        # data MVAQD
        "db_name": "MVAQD",  # database name [ PIPAL | LIVE | CSIQ | TID2013 ]
        "db_path": "E:\\ImageDatabase\\MVAQD\\MVAQD_315",  # root of dataset
        "snap_path": "./weights/CVIQ",  # path for saving weights
        "txt_file_name": "./IQA_list/MVAQD.txt",  # image list file (.txt)
        "train_size": 0.8,
        "scenes": list(range(315)),

        # ensemble in validation phase
        #"test_ensemble": False, # for Live3D database
        "test_ensemble": True,
        "n_ensemble": 5,

        # optimization
        "batch_size": 8,
        #"batch_size": 7,
        # "learning_rate": 2e-4,学习率
        "learning_rate": 2e-5,
        "weight_decay": 1e-5,
        "n_epoch": 100,
        "val_freq": 1,
        "save_freq": 5,
        "checkpoint": None,  # load pretrained weights
        "T_max": 50,  # cosine learning rate period (iteration)
        "eta_min": 0  # mininum learning rate
    })
    return config


class RandCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        d_img = sample['d_img']
        score = sample['score']

        c, h, w = d_img.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        d_img = d_img[:, top: top+new_h, left: left+new_w]

        sample = {'d_img': d_img, 'score': score}
        return sample
    

class RandHorizontalFlip(object):
    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img']
        score = sample['score']

        prob_lr = np.random.random()
        # np.fliplr needs HxWxC -> transpose from CxHxW to HxWxC
        # after the flip ends, return to CxHxW
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img.transpose((1, 2, 0))).copy().transpose((2, 0, 1))

        sample = {'d_img': d_img, 'score': score}
        return sample


class RandRotation(object):
    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img']
        score = sample['score']

        prob_rot = np.random.uniform()

        if prob_rot < 0.25:     # rot0
            pass
        elif prob_rot < 0.5:    # rot90
            d_img = np.rot90(d_img.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
        elif prob_rot < 0.75:   # rot180
            d_img = np.rot90(d_img.transpose((1, 2, 0)), 2).copy().transpose((2, 0, 1))
        else:                   # rot270
            d_img = np.rot90(d_img.transpose((1, 2, 0)), 3).copy().transpose((2, 0, 1))
        
        sample = {'d_img': d_img, 'score': score}
        return sample
        


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img']
        score = sample['score']

        d_img = (d_img - self.mean) / self.var

        sample = {'d_img': d_img, 'score': score}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        # r_img: C x H x W (numpy->tensor)
        d_img = sample['d_img']
        score = sample['score']

        d_img = torch.from_numpy(d_img)
        score = torch.from_numpy(score)

        sample = {'d_img': d_img, 'score': score}
        return sample


def RandShuffle(scenes, train_size=0.8):
    #if scenes == "all":
     #   scenes = list(range(15)) # [0,1,2...15]
    
    n_scenes = len(scenes) # 16
    n_train_scenes = int(np.floor(n_scenes * train_size)) # 16*0.8取整=12
    n_test_scenes = n_scenes - n_train_scenes # 4

    seed = np.random.random() # 返回[0,1)之间的一个小数，比如0.12
    # 下面这两行是做什么的
    random_seed = int(seed*10) # 比如1
    np.random.seed(random_seed)
    np.random.shuffle(scenes) # 打乱[0，1，2。。。15]的排序，比如[10,5,3,...2]
    train_scene_list = scenes[:n_train_scenes] # scenes的前12个
    test_scene_list = scenes[n_train_scenes:] # scenes的后4个
    
    return train_scene_list, test_scene_list