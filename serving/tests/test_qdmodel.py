# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  models
   Author :       machinelp
   Date :         2020-08-27
-------------------------------------------------

'''

import os
import sys
import time
import json
import argparse
import numpy as np
from core.models import QDNetModel
from qdnet.conf.config import load_yaml

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
parser.add_argument('--img_path', help='config file path')
parser.add_argument('--fold', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)

if __name__ == '__main__':

    qd_model = QDNetModel(config, args.fold)
    pre = qd_model.predict(args.img_path)
    print (">>>>>", pre) 

