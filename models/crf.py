from __future__ import print_function
from cgi import print_environ
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
import numpy as np
from torch.autograd import Variable
import logging
import os
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import logging
from math import exp
from collections import OrderedDict, namedtuple
from torch.nn import init

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, same_padding=True,  bias=True):
        super(Conv2d, self).__init__()

        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,  padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        #if self.bn is not None:
        x = self.bn(x)
        #if self.relu is not None:
        x = self.relu(x)
        return x
    
class MessagePassing(nn.Module):
    def __init__(self,branch_n, input_ncs, bn=False):
        super(MessagePassing, self).__init__()
        self.branch_n = branch_n
        self.iters = 1 # 迭代次数
        self.input_ncs = input_ncs       
        
        for i in range(branch_n):
            for j in range(branch_n):
                if i == j: 
                    continue
                setattr(self, "w_0_{}_{}_0".format(j, i), 
                nn.Sequential(
                    nn.Conv2d(input_ncs[j], input_ncs[i], kernel_size=1,  stride=1) # 根据需求调整 padding                   
                    )
                        )
        self.relu = nn.ReLU(inplace=True) #非原位ReLUctant激活函数
        self.prelu = nn.PReLU() #参数ReLU激活函数
        
    def forward(self, input):
        hidden_state = input
        side_state = []

        for _ in range(self.iters): #迭代次数
            hidden_state_new = []
            for i in range(self.branch_n): #对每一个分支
                
                unary = hidden_state[i] #获取当前分支的维度
                binary = None
                for j in range(self.branch_n): #
                    if i == j: #如果分支相同，跳过此次循环
                        continue
                    if binary is None:
                        binary = getattr(self, 'w_0_{}_{}_0'.format(j, i))(hidden_state[j]) #获取属性，执行卷积层，并将当前分支j的hidden作为输入进行卷积运算

                    else:
                        binary = binary + getattr(self, 'w_0_{}_{}_0'.format(j, i))(hidden_state[j]) #获取属性，执行卷积

                binary = self.relu(binary) #执行参数ReLU激活
                hidden_state_new += [self.relu(unary +binary)] #计算新的隐藏状态
                #hidden_state_new += unary + binary
                
            hidden_state = hidden_state_new
        return hidden_state
    

class MFuses(nn.Module):
    def __init__(self, out_dim, bn=False):
        super(MFuses, self).__init__()

        self.out_dim = out_dim     
        
        self.decoder1 = nn.Sequential(
               
               nn.Conv2d(256,128, kernel_size=3, stride=2, padding=1, bias=False),                         
                                   nn.BatchNorm2d(128),
                                   nn.ReLU()
            )
        self.decoder2 = nn.Sequential(
               
               nn.Conv2d(512,256, kernel_size=3, stride=2, padding=1, bias=False),                         
                                   nn.BatchNorm2d(256),
                                   nn.ReLU()
            )
        self.decoder3 = nn.Sequential(
               
                nn.Conv2d(1024,512, kernel_size=3, stride=2, padding=1, bias=False),                         
                                   nn.BatchNorm2d(512),
                                   nn.ReLU()
            )
        
        
    def forward(self, x1,x2):
        print(x1.size(),x2.size())
        x = torch.cat([x1,x2],dim=1)
        b,c,h,w = x.shape
        if c==256:
            x_d = self.decoder1(x) #256           
            #print(x_l.size())
            x_t = torch.stack((x1,x2),dim=1)
            x_t =torch.mean(x_t,dim=1)
            
            x =x_t+x_d
            #print(x_t.size())
        elif c==512:
            x_d = self.decoder2(x)
            #print(x_d.size())
            x_t = torch.stack((x1,x2),dim=1)
            x_t =torch.mean(x_t,dim=1)
            
            x =x_t+x_d
        else: 
            x_d = self.decoder3(x)
            #print(x_d.size())
            x_t = torch.stack((x1,x2),dim=1)
            x_t =torch.mean(x_t,dim=1)
            
            x =x_t+x_d
       
        b_f, c, h,w =x.shape
        x=x.view(b_f//17, 17,c)
        #print(x.size())
                     
        return x

class CRF(nn.Module):
    def __init__(self, branch_n=2,output_stride=1, bn=False, *args, **kwargs):
        super(CRF, self).__init__()
        self.branch_n = 2
        self.output_stride = output_stride
        self.passing1 = MessagePassing(branch_n=2,
                                       input_ncs=[128,128])
        self.passing2 = MessagePassing(branch_n=2,
                                       input_ncs=[256,256])
        self.passing3 = MessagePassing(branch_n=2,
                                       input_ncs=[512,512])
        self.mfuser1 = MFuses(
                             out_dim=128)
        self.mfuser2 = MFuses(
                             out_dim=256)
        self.mfuser3 = MFuses(
                             out_dim=512)
      
        

    def forward(self, x1, x2):
        #print(x1.size(),x2.size())
        b,c ,h,w = x1.shape
        
        if c == 128:
            #x1 =self.conv1(x1)
            x1, x2 = self.passing1([x1,x2])
           
            #x = self.mfuser1(x1,x2)
            #x=x1+x2
        elif c == 256:
            #print(1)
            #x2=self.conv2(x2)
            x1, x2  = self.passing2([x1,x2])
            #x = self.mfuser2(x1,x2)
            #x = x1 + x2
        else:
            x1, x2  = self.passing3([x1,x2])
            #x = self.mfuser3(x1,x2)
            #x = x1 + x2
        x_t = torch.stack((x1, x2), dim=1)
        x = torch.mean(x_t, dim=1)
        b_f, c, h, w = x.shape
        x = x.view(b_f // 17, 17, c)
        return x


def crf():
    model = CRF().to('cuda:0')
    return model

if __name__ == '__main__':
    img = torch.randn((2, 32, 3, 112, 112))
    model = CRF().to('cuda:0')
    model(img)