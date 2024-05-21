import torch
import torch.nn as nn
from . import helpers
from .base_loss import BaseLoss


class LossCIP(BaseLoss):
    """
    Loss + Conditional Invariance Penalty (CIP). CIP is default to linear mean difference.
    """
    def __init__(self, device, nb_classes, lamCIP, discrepType='mean', loss_type='CrossEntropyLoss'):
        super().__init__(device)
        self.nb_classes = nb_classes
        self.lamCIP = lamCIP
        self.discrepType = discrepType
        self.criterion = getattr(nn, loss_type)

        if self.discrepType == 'MMD':
            self.discrepancy = helpers._rbf_mmd2
        else: # self.discrepType = 'mean':
            self.discrepancy = helpers._diff_mean
            
            
    def cip_penalty(self, feats, y, groups=None):
        penalty = 0.
        if groups is None:
            nb_envs = len(feats)
            for i1 in range(nb_envs):
                for i2 in range(nb_envs):
                    if i1 > i2:
                        for i in range(self.nb_classes):
                            n1 = torch.sum(y[i1]==i)
                            n2 = torch.sum(y[i2]==i)
                            if n1 > 0 and n2 > 0:
                                penalty += self.discrepancy(feats[i1][y[i1]==i], 
                                                            feats[i2][y[i2]==i])
            
        else:
            group_ids = torch.unique(groups)
            nb_envs = len(group_ids)
            for id1 in group_ids:
                for id2 in group_ids:
                    if id1 > id2:
                        for i in range(self.nb_classes):
                            n1 = torch.sum(y[groups==id1]==i)
                            n2 = torch.sum(y[groups==id2]==i)
                            if n1 > 0 and n2 > 0:
                                penalty += self.discrepancy(feats[groups==id1][y[groups==id1]==i], 
                                                            feats[groups==id2][y[groups==id2]==i])
        
        return penalty / (nb_envs ** 2) * 2
    
    
    def __call__(self, data, model, groups=None, **kwargs):
        
        if groups is None:
            nb_envs = len(data)
            feats = [None] * nb_envs
            ys = [None] * nb_envs
            loss = 0.
            for i in range(nb_envs):
                x, y = data[i][0].to(self.device), data[i][1].to(self.device)
                feats[i] = model.featurizer(x)
                ys[i] = y
                outputs = model.classifier(feats[i])
                loss += self.criterion(**kwargs)(outputs, y)
            loss /= nb_envs
            penalty = self.cip_penalty(feats, ys)
            loss_penalty = loss + self.lamCIP * penalty
            return loss_penalty
        
        else:
            
            x, y = data[0][0].to(self.device), data[0][1].to(self.device)
            feats = model.featurizer(x)
            outputs = model.classifier(feats)
            
            loss = self.criterion(**kwargs)(outputs, y)
            penalty = self.cip_penalty(feats, y, groups)
            
            loss_penalty = loss + self.lamCIP * penalty
        
            return loss_penalty