import torch
import torch.nn as nn
from . import helpers
from .base_loss import BaseLoss


class LossDIP(BaseLoss):
    """
    Loss + Domain invariant penalty (DIP). DIP is default to linear mean difference.
    """
    def __init__(self, device, nb_classes, lamDIP, discrepType='mean', loss_type='CrossEntropyLoss', freeze_bn=False):
        super().__init__(device)
        self.lamDIP = lamDIP
        self.nb_classes = nb_classes
        self.discrepType = discrepType
        self.freeze_bn = freeze_bn
        
        if self.discrepType == 'MMD':
            self.discrepancy = helpers._rbf_mmd2
        else: # self.discrepType = 'mean':
            self.discrepancy = helpers._diff_mean
        
        self.criterion = getattr(nn, loss_type)



    def dip_penalty(self, source_feats, target_feats, source_y=None, **kwargs):
        
        if 'weight' not in kwargs:
            penalty = self.discrepancy(source_feats, target_feats)
        else:
            # need to weigh the data points in the DIP penalty
            source_y = source_y.to(self.device)
            labels_onehot = torch.zeros((source_y.shape[0], self.nb_classes), dtype=torch.float).to(self.device)
            labels_onehot = labels_onehot.scatter_(1, source_y.reshape(-1, 1), 1)
            sample_weight = labels_onehot.matmul(kwargs['weight'])
            penalty = self.discrepancy(source_feats, target_feats, sample_weight=sample_weight)

        return penalty
    
    
    def __call__(self, data, model, groups=None, **kwargs):
        
        
        if groups is None:
            target_x = data[-1][0].to(self.device)
            target_feats = model.featurizer(target_x)
            
            nb_envs = len(data)-1
            loss = 0.
            penalty = 0.
            
            for i in range(nb_envs):
                source_x, source_y = data[i][0].to(self.device), data[i][1].to(self.device)
                source_feats = model.featurizer(source_x)
                source_outputs = model.classifier(source_feats)
                loss += self.criterion(**kwargs)(source_outputs, source_y)
                penalty += self.dip_penalty(source_feats, target_feats, source_y, **kwargs)
            
            loss_penalty = loss + self.lamDIP * penalty
            return loss_penalty / nb_envs
        
        else:
            
            source_x, source_y = data[0][0].to(self.device), data[0][1].to(self.device)
            target_x = data[-1][0].to(self.device)  
            
            
            if self.freeze_bn:
                source_feats = model.featurizer(source_x)
                helpers.freeze_bn_layers(model)
                target_feats = model.featurizer(target_x)
                helpers.unfreeze_bn_layers(model)
            
            else:
                batch_size = source_x.shape[0]
                x = torch.cat([source_x, target_x], dim=0)
                feats = model.featurizer(x)
                source_feats, target_feats = feats[:batch_size], feats[batch_size:]
             
            source_outputs = model.classifier(source_feats)
            
            loss = self.criterion(**kwargs)(source_outputs, source_y)
            penalty = self.dip_penalty(source_feats, target_feats, source_y, **kwargs)
            
            loss_penalty = loss + self.lamDIP * penalty
            return loss_penalty