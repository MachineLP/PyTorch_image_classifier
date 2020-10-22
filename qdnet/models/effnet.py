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
from qdnet.models.metric_strategy import Swish_module, ArcMarginProduct_subcenter, ArcFaceLossAdaptiveMargin


class Effnet(nn.Module):
    '''
    '''
    def __init__(self, enet_type, out_dim, drop_nums=1, pretrained=False, metric_strategy=False):
        super(Effnet, self).__init__()
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(drop_nums) ])
        in_ch = self.enet.classifier.in_features
        self.fc = nn.Linear(in_ch, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.classify = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()
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
