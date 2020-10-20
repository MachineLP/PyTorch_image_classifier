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
from resnest.torch import resnest50
from resnest.torch import resnest101
from resnest.torch import resnest200
from resnest.torch import resnest269


class Resnest(nn.Module):
    '''
    '''
    def __init__(self, enet_type, out_dim, drop_nums=1, pretrained=False):
        super(Resnest, self).__init__()
        if enet_type in ["resnest50", "resnest101", "resnest200", "resnest269"]:
            self.enet = locals()[enet_type](pretrained=pretrained)
        self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(drop_nums) ])
        in_ch = self.enet.fc.in_features
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
 