# -*- coding:utf-8 -*-

import os
import apex
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from apex import amp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from qdnet.conf.config import load_yaml
from qdnet.optimizer.optimizer import GradualWarmupSchedulerV2
from qdnet.dataset.dataset import get_df, QDDataset
from qdnet.dataaug.dataaug import get_transforms
from qdnet.models.effnet import Effnet
from qdnet.models.resnest import Resnest
from qdnet.models.se_resnext import SeResnext
from qdnet.loss.loss import Loss
from qdnet.conf.constant import Constant


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:

        optimizer.zero_grad()
        
        data, target = data.to(device), target.to(device)

        loss = Loss(out_dim=int(config["out_dim"]), loss_type=config["loss_type"])(model, data, target, mixup_cutmix=config["mixup_cutmix"])

        if not config["use_amp"]:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if int(config["image_size"]) in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img):
   return img


def val_epoch(model, loader, mel_idx, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = torch.zeros((data.shape[0], int(config["out_dim"]))).to(device)
            probs = torch.zeros((data.shape[0], int(config["out_dim"]))).to(device)

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = Loss(out_dim=int(config["out_dim"]), loss_type=config["loss_type"])(model, data, target, mixup_cutmix=False)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
        return val_loss, acc, auc


def run(fold, df, transforms_train, transforms_val, mel_idx):

    df_train = df[df['fold'] != fold]
    df_valid = df[df['fold'] == fold]

    dataset_train = QDDataset(df_train, 'train', transform=transforms_train)
    dataset_valid = QDDataset(df_valid, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=int(config["batch_size"]), sampler=RandomSampler(dataset_train), num_workers=int(config["num_workers"]))
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=int(config["batch_size"]), num_workers=int(config["num_workers"]))

    model = ModelClass(
        enet_type = config["enet_type"],
        out_dim = int(config["out_dim"]),
        drop_nums = int(config["drop_nums"]),
        pretrained = config["pretrained"],
        margin_strategy = config["metric_strategy"]
    )
    if DP:
        model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)

    auc_max = 0.  
    model_file  = os.path.join(config["model_dir"], f'best_fold{fold}.pth')
    model_file3 = os.path.join(config["model_dir"], f'final_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=float(config["init_lr"]))
    if config["use_amp"]: 
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)
#     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config["n_epochs"]) - 1) 
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(config["n_epochs"]) - 1) 
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, int(config["n_epochs"]) + 1): 
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')
#         scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc = val_epoch(model, valid_loader, mel_idx)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}.'
        print(content)  
        with open(os.path.join(config["log_dir"], f'log.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()    
        if epoch==2: scheduler_warmup.step() # bug workaround   
            
        if auc > auc_max:
            print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            torch.save(model.state_dict(), model_file)
            auc_max = auc

    torch.save(model.state_dict(), model_file3)


def main():

    df, df_test, mel_idx = get_df( config["data_dir"], config["auc_index"]  )

    transforms_train, transforms_val = get_transforms(config["image_size"])  

    folds = [int(i) for i in config["fold"].split(',')]  
    for fold in folds:
        run(fold, df, transforms_train, transforms_val, mel_idx)


if __name__ == '__main__':

    os.makedirs(config["model_dir"], exist_ok=True)  
    os.makedirs(config["log_dir"], exist_ok=True)    
    os.environ['CUDA_VISIBLE_DEVICES'] = config["CUDA_VISIBLE_DEVICES"]

    if config["enet_type"] in Constant.RESNEST_LIST:   
        ModelClass = Resnest
    elif config["enet_type"] in Constant.SERESNEXT_LIST:
        ModelClass = SeResnext
    elif config["enet_type"] in Constant.GEFFNET_LIST:
        ModelClass = Effnet
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed()

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()

