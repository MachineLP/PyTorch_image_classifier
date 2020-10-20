# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QDNet Model effnet
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------
'''



import torch
import torch.nn as nn
from pretrainedmodels import se_resnext101_32x4d


class SeResnext(nn.Module):
    '''
    '''
    def __init__(self, enet_type, out_dim, drop_nums=1, pretrained=False):
        super(SeResnext, self).__init__()
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(drop_nums) ])
        in_ch = self.enet.last_linear.in_features
        self.fc = nn.Linear(in_ch, out_dim)
        self.enet.fc = nn.Identity()
   
    def extract(self, x):
        x = self.enet(x)
        return x
    
    def forward(self, x):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        out /= len(self.dropouts)
        return out
