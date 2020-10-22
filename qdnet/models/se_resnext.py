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
from qdnet.models.metric_strategy import Swish_module, ArcMarginProduct_subcenter, ArcFaceLossAdaptiveMargin


class SeResnext(nn.Module):
    '''
    '''
    def __init__(self, enet_type, out_dim, drop_nums=1, pretrained=False, metric_strategy=False):
        super(SeResnext, self).__init__()
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(drop_nums) ])
        in_ch = self.enet.last_linear.in_features
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
