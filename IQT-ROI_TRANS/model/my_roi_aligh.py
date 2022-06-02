import math

import scipy.io
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision as tv
from torchvision.ops import RoIPool, RoIAlign
import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        #sz: B,512,10,10 =>1.MAX: B,512,1,1 |||2.MEAN:B,512,1,1
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1) #把平均池化和最大池化在维度1拼接


def get_idx(batch_size, n_output, device=None):
    idx = torch.arange(float(batch_size), dtype=torch.float, device=device).view(1, -1)# 比如batch_size=8,arange(batch_size)=tensor([0,1,2,...,7]),是一维的，view(1,-1)得到tensor([[0,1,2,...,7]])
    idx = idx.repeat(n_output, 1, ).t()# n_output是一个图像被分的块数，比如100，repeat(第一个维度复制n_output次，第二个维度不变)[[0~7],[0~7],...,[0~7]]100行8列,t()是转置，变为8行100列[[0~0],[1~1],...,[7~7]]
    idx = idx.contiguous().view(-1, 1)#contiguous()返回一个内存连续的有相同数据的 tensor,用于view()前,view(-1, 1)变为800行1列，[[0],...[0],[1],...,[1],...,[7]]
    return idx


def get_blockwise_rois_old(blk_size, img_size=None, batch_size=0): #用来得到每个块的位置信息，一个块用左上和右下的两个坐标表示，blk_size块的个数，比如[10,10]
    if img_size is None: img_size = [1, 1]
    y = np.linspace(0, img_size[0], num=blk_size[0] + 1) #y==>H 序列生成器linspace(起点0，终点1024，个数11) [0 102.4 204.8 .。。1024 ]
    x = np.linspace(0, img_size[1], num=blk_size[1] + 1) #x==>W [0 204.8 .。。2048 ]
    a = []
    for n in range(len(y) - 1):
        for m in range(len(x) - 1):
            a += [x[m], y[n], x[m + 1], y[n + 1]] #从左到右从上到下扫描 [0 0 204.8 102.4 204.8 0 409.6 102.4...]
    a = torch.tensor(a).view(1, -1) #tensor([[0 0 204.8 102.4 204.8 0 409.6 102.4...]]) (1,400)
    a = a.repeat(1, batch_size).t() #(400*batch_size,1)
    a = a.contiguous().view(-1, 4) #(100*batch_size,4) tensor([0 0 204.8 102.4],[204.8 0 409.6 102.4],...,[0 0 204.8 102.4],...)
    return a

def get_blockwise_rois(batch_size=0): #用来得到每个块的位置信息，一个块用左上和右下的两个坐标表示，blk_size块的个数，比如[10,10]
    a = scipy.io.loadmat('E:\\mxy\\IQT-ROI_TRANS\\model\\loc_map.mat')['a']
    #a = np.array([0, 0, 2672, 1336])
    a = torch.tensor(a).view(1, -1) #tensor([[0 0 204.8 102.4 204.8 0 409.6 102.4...]]) (1,400)
    a = a.repeat(1, batch_size).t() #(400*batch_size,1)
    a = a.contiguous().view(-1, 4) #(100*batch_size,4) tensor([0 0 204.8 102.4],[204.8 0 409.6 102.4],...,[0 0 204.8 102.4],...)
    return a

def getROITrans():
    model = RoIPoolTrans()
    checkpoints = torch.load('E:\WorkSpacePython\paq2piq\weights\\modelw_E006.pth')
    pretrained_dict = checkpoints['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class RoIPoolTrans(nn.Module):
    rois = None
    criterion = nn.MSELoss()

    def __init__(self, backbone='resnet18'):
        super().__init__()
        if backbone is 'resnet18':
            model = tv.models.resnet18(pretrained=True)
            cut = -2
            spatial_scale = 1 / 32

        self.model_type = self.__class__.__name__
        self.body = nn.Sequential(*list(model.children())[:cut]) #从reset18第一层到倒数第二层
        self.roi_pool = RoIAlign((10, 10), spatial_scale=spatial_scale, sampling_ratio=4) #tv里的函数，一张图像输出维度为(10,10)
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),#(B*100,1024,1,1)
            nn.Flatten(),#(B*100,1024)
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(in_features=1024, out_features=512, bias=True),#(B*100,512)
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=256, bias=True)#(B*100,256)
        )
    def forward(self, im_data): # im_data (B,W,H)
        feats = self.body(im_data)  # 经过resnet18处理后(B,512,W/32,H/32)
        batch_size = im_data.size(0) # B
        #idx = get_idx(batch_size, 100, im_data.device) # (B*100,1) 其中100是10*10得到的
        rois_data = get_blockwise_rois(batch_size).float().to(im_data.device)  # (100*B,4)
        idx = get_idx(batch_size, rois_data.size(0)//batch_size, im_data.device)  # (B*100,1) 其中100是10*10得到的

        indexed_rois = torch.cat((idx, rois_data), 1) # (100*B,5)
        features = self.roi_pool(feats, indexed_rois) #(B*100,512,10,10)
        features = self.head(features).view(batch_size, rois_data.size(0)//batch_size, 256) #(B,100,256) 一副图像最终变成100个token，每个token的维度为256
        return features


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50_backbone(**kwargs):
    # Constructs a ResNet-50 model_hyper.
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)

    # load pre-trained weights
    # save_model = model_zoo.load_url(model_urls['resnet50'])
    save_model = torch.load('./model/resnet50.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    pass