
import torch
import numpy as np
import torch.nn as nn
from qdnet.loss.focal_loss import FocalLoss

class Loss(object):
    def __init__(self, loss_type="ce", w=None):
        # w=torch.tensor([10,2,15,20],dtype=torch.float)  
        if loss_type == "ce_loss":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif loss_type == "focal_loss":
            device = torch.device('cuda')
            self.criterion = FocalLoss().to(device)
        elif loss_type == "bce_loss":
            self.criterion = nn.BCEWithLogitsLoss(w)
        elif loss_type == "mlsm_loss":
            self.criterion = nn.MultiLabelSoftMarginLoss(w)  
        else:
            raise NotImplementedError()
    
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # self.criterion = FocalLoss().to(self.device)

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


    def cutmix(self, data, targets, alpha):
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
    
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

        targets = [targets, shuffled_targets, lam]
        return data, targets


    def cutmix_criterion(self, preds, targets):
        targets1, targets2, lam = targets[0], targets[1], targets[2]
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)


    def mixup(self, data, targets, alpha):
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
    
        lam = np.random.beta(alpha, alpha)
        data = data * lam + shuffled_data * (1 - lam)
        targets1 = [targets, shuffled_targets, lam]
        return data, targets1


    def mixup_criterion(self, preds, targets):
        targets1, targets2, lam = targets[0], targets[1], targets[2]
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)

    
    def __call__(self, model, images, targets, mixup_cutmix=False, alpha1=0.2, alpha2=1.0):
        if mixup_cutmix:
            if np.random.rand()<0.5:
                with torch.no_grad():
                    images_mixup, targets_mixup = self.mixup(images, targets, alpha1)

                outputs_mixup = model(images_mixup)
                loss = self.mixup_criterion(outputs_mixup, targets_mixup) 

            else: 
                with torch.no_grad():
                    images_cutmix, targets_cutmix = self.cutmix(images, targets, alpha2)
                outputs_cutmix = model(images_cutmix)
                loss = self.cutmix_criterion(outputs_cutmix, targets_cutmix) 
        
        else:
            outputs = model(images)
            loss = self.criterion(outputs, targets)
        
        return loss

