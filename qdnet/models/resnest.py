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
from qdnet.models.metric_strategy import Swish_module, ArcMarginProduct_subcenter, ArcFaceLossAdaptiveMargin


class Resnest(nn.Module):
    '''
    '''
    def __init__(self, enet_type, out_dim, drop_nums=1, pretrained=False, metric_strategy=False):
        super(Resnest, self).__init__()
        if enet_type in ["resnest50", "resnest101", "resnest200", "resnest269"]:
            self.enet = locals()[enet_type](pretrained=pretrained)
        self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(drop_nums) ])
        in_ch = self.enet.fc.in_features
        self.fc = nn.Linear(in_ch, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.classify = nn.Linear(in_ch, out_dim)
        self.enet.fc = nn.Identity()
        self.metric_strategy = metric_strategy
   
    def extract(self, x):
        x = self.enet(x)
        return x
   
    def forward(self, x):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.metric_strategy:
             out = self.metric_classify(self.swish(self.fc(x)))
        else:
             for i, dropout in enumerate(self.dropouts):
                 if i == 0:
                     out = self.classify(dropout(x))
                 else:
                     out += self.classify(dropout(x))
             out /= len(self.dropouts)
        return out
 