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
import geffnet


class Effnet(nn.Module):
    '''
    '''
    def __init__(self, enet_type, out_dim, drop_nums=1, pretrained=False):
       super(Effnet, self).__init__()
       self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
       self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(drop_nums) ])
       in_ch = self.enet.classifier.in_features
       self.fc = nn.Linear(in_ch, out_dim)
       self.enet.classifier = nn.Identity()
   
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
