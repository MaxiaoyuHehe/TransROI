import os

import scipy.io
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr


""" train model """
def train_epoch(config, epoch, model_transformer, criterion, optimizer, scheduler, train_loader):
    losses = []
    model_transformer.train()
    # value is not changed
    enc_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device) #tensor 16*289 全1

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    for data in tqdm(train_loader): #对于train——loader里的所有数据，一次是一个batch的数据，for循环结束则一个epoch结束
        # labels: batch size 
        # enc_inputs: batch_size x len_seq+1
        # enc_inputs_embed: batch_size x len_seq x n_feats
            
        d_img = data['d_img'].to(config.device) #16*3*384*768
        labels = data['score'] #16*1
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device) #16
        # weight update
        optimizer.zero_grad()
        pred = model_transformer(enc_inputs, d_img) # 16*1
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item() # float
        losses.append(loss_val)

        loss.backward() #误差后向传播
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy() # np数组 16*1
        labels_batch_numpy = labels.data.cpu().numpy() # np数组 16
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch)) #float 当前epoch计算得到的srcc值
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

    return np.mean(losses), rho_s, rho_p # 返回这个epoch平均的loss和srcc以及prcc


""" validation """
def eval_epoch(config, epoch, model_transformer, criterion, test_loader):
    with torch.no_grad(): #不更新梯度
        losses = []
        model_transformer.eval()

        # value is not changed
        enc_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            # labels: batch size 
            # enc_inputs: batch_size x len_seq / dec_inputs: batch_size x len_seq
            # enc_inputs_embed: batch_size x len_seq x n_feats / dec_inputs_embed: batch_size x len_seq x n_feats

            d_img = data['d_img']  # 16*3*512*1024
            labels = data['score']  # 16*1
            labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)  # 16
            if config.test_ensemble:    # use test ensemble
                pred = 0
                for i in range(config.n_ensemble): #每个batch的图片分别随机裁切5次所得结果取平均
                    b, c, h, w = d_img.size()
                    new_h, new_w = config.crop_size # new_h=384 new_w=768
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w) #这两行用来确定裁切图片左上角的点
                    d_img_crop = d_img[:, :, top: top+new_h, left: left+new_w].to(config.device) # 16*3*384*768
                    pred += model_transformer(enc_inputs, d_img_crop) # 将所有集成的结果相加
                
                pred = pred / config.n_ensemble # 16*1 平均值
            else:
                d_img=d_img.to(config.device)
                pred = model_transformer(enc_inputs, d_img)




            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            loss_val = loss.item()
            losses.append(loss_val)

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy() #np数组 16*1
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy) #每个batch的值都存储起来，循环结束得到一个epoch的值
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

        scipy.io.savemat('./results/pred%03d.mat'%epoch, {'pred':pred_epoch})
        scipy.io.savemat('./results/gt.mat', {'gt': labels_epoch})
        print(labels_epoch.shape)

        return np.mean(losses), rho_s, rho_p
