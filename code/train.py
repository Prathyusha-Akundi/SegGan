#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from net import NetS, NetC
from dataloader_segan import Dataset, loader


# In[16]:


batchSize = 36
niter = 5000
lr = 0.002
beta1 = 0.5
decay = 0.5
seed = 666
outpath = '/scratch/prathyuakundi/MIA_data/outputs'
outpath_home = '/home/prathyuakundi/MIA/Project/outpaths_final'


# In[3]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[4]:


def dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total


# In[5]:


torch.manual_seed(seed)


# In[6]:


torch.cuda.manual_seed(seed)


# In[7]:


cudnn.benchmark = True


# In[8]:


print('===> Building model')
segmentor = NetS()
# NetS.apply(weights_init)
print(segmentor)
critic = NetC()
# NetC.apply(weights_init)
print(critic)


# In[9]:


segmentor = segmentor.cuda()
critic = critic.cuda()


# In[10]:


optimizerG = optim.Adam(segmentor.parameters(), lr=lr, betas=(beta1, 0.999))


# In[11]:


optimizerD = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))


# In[12]:


dataloader = loader(Dataset('./'),batchSize)


# In[13]:


max_iou = 0
segmentor.train()

for epoch in range(niter):
    for i, data in enumerate(dataloader, 1):
        #train C
        critic.zero_grad()
        input_tensor, label = Variable(data[0]), Variable(data[1])
        
        input_tensor = input_tensor.cuda()
        target = label.cuda()
        
        target = target.type(torch.FloatTensor)
        target = target.cuda()
        
        output = segmentor(input_tensor)
        #output = F.sigmoid(output*k)
        output = F.sigmoid(output)
        output = output.detach()
        output_masked = input_tensor.clone()
        input_mask = input_tensor.clone()
        
        #detach G from the network
        for d in range(3):
            output_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * output).squeeze()
        
        output_masked = output_masked.cuda()
        result = critic(output_masked)
        target_masked = input_tensor.clone()
        for d in range(3):
            target_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * target).squeeze()
        
        target_masked = target_masked.cuda()
        target_D = critic(target_masked)
        loss_D = - torch.mean(torch.abs(result - target_D))
        loss_D.backward()
        optimizerD.step()
        #clip parameters in D
        for p in critic.parameters():
            p.data.clamp_(-0.05, 0.05)
        #train G
        segmentor.zero_grad()
        output = segmentor(input_tensor)
        output = F.sigmoid(output)

        for d in range(3):
            output_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * output).squeeze()
        
        output_masked = output_masked.cuda()
        result = critic(output_masked)
        for d in range(3):
            target_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * target).squeeze()
        
        target_masked = target_masked.cuda()
        target_G = critic(target_masked)
        loss_dice = dice_loss(output,target)
        loss_G = torch.mean(torch.abs(result - target_G))
        loss_G_joint = torch.mean(torch.abs(result - target_G)) + loss_dice
        loss_G_joint.backward()
        optimizerG.step()

    print("===> Epoch[{}]({}/{}): Batch Dice: {:.4f}".format(epoch, i, len(dataloader), 1 - loss_dice.data))
    print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_G.data))
    print("===> Epoch[{}]({}/{}): D_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_D.data))


    if(epoch % 200 == 0):
        vutils.save_image(data[0],'{0}/input_{1}.png'.format(outpath_home,epoch),normalize=True)
        vutils.save_image(data[1], '{0}/label_{1}.png'.format(outpath_home,epoch),normalize=True)
        vutils.save_image(output.data, '{0}/result_{1}.png'.format(outpath_home,epoch),normalize=True)

    
    if(epoch % 100 == 0):
        vutils.save_image(data[0],'{0}/input_{1}.png'.format(outpath,epoch),normalize=True)
        vutils.save_image(data[1], '{0}/label_{1}.png'.format(outpath,epoch),normalize=True)
        vutils.save_image(output.data, '{0}/result_{1}.png'.format(outpath,epoch),normalize=True)
    
   
        segmentor.eval()
        IoUs, dices = [], []
        for i, data in enumerate(dataloader, 1):
            input_val, gt_val = Variable(data[0]), Variable(data[1])
            
            input_val = input_val.cuda()
            gt_val = gt_val.cuda()
            pred = segmentor(input_val)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.type(torch.LongTensor)
            pred_np = pred.data.cpu().numpy()
            gt_val = gt_val.data.cpu().numpy()
            for x in range(input_val.size()[0]):
                IoU = np.sum(pred_np[x][gt_val[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt_val[x]) - np.sum(pred_np[x][gt_val[x]==1]))
                dice = np.sum(pred_np[x][gt_val[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt_val[x]))
                IoUs.append(IoU)
                dices.append(dice)

        
        segmentor.train()
        IoUs = np.array(IoUs, dtype=np.float64)
        dices = np.array(dices, dtype=np.float64)
        mIoU = np.mean(IoUs, axis=0)
        mdice = np.mean(dices, axis=0)
        print('mIoU: {:.4f}'.format(mIoU))
        print('Dice: {:.4f}'.format(mdice))
        if mIoU > max_iou:
            max_iou = mIoU
            torch.save(segmentor.state_dict(), '%s/segmentor_epoch_%d.pth' % (outpath, epoch))
        vutils.save_image(data[0],
                '%s/input_val.png' % outpath,
                normalize=True)
        vutils.save_image(data[1],
                '%s/label_val.png' % outpath,
                normalize=True)
        pred = pred.type(torch.FloatTensor)
        vutils.save_image(pred.data,
                '%s/result_val.png' % outpath,
                normalize=True)
    if epoch % 25 == 0:
        lr = lr*decay
        if lr <= 0.00000001:
            lr = 0.00000001
        print('Learning Rate: {:.6f}'.format(lr))
        # print('K: {:.4f}'.format(k))
        print('Max mIoU: {:.4f}'.format(max_iou))
        optimizerG = optim.Adam(segmentor.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerD = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))


# In[15]:



# In[ ]:




