import torch
import torch.nn as nn
from .base_loss import BaseLoss


class LossgroupDRO(BaseLoss):
    """
    The groupDRO loss
    """
    def __init__(self, device, group_weights_lr, n_domains, loss_type='CrossEntropyLoss'):
        super().__init__(device)
        self.group_weights_lr = group_weights_lr
        self.group_weights = None
        self.criterion = getattr(nn, loss_type)
        
        self.n_domains = n_domains
        self.group_weights = torch.ones(self.n_domains).to(self.device)
        self.group_weights = self.group_weights/self.group_weights.sum()

    def __call__(self, data, model, groups=None, **kwargs):
        
        if groups is None:
            
            losses = torch.zeros(self.n_domains).to(self.device)
            for i in range(len(data)):
                x, y = data[i][0].to(self.device), data[i][1].to(self.device)
                outputs = model(x)
                losses[i] = self.criterion(reduction='mean', **kwargs)(outputs, y)

        
        else:
            losses = torch.zeros(self.n_domains).to(self.device)
            group_ids = torch.unique(groups)
            
            x, y = data[0][0].to(self.device), data[0][1].to(self.device)
            outputs = model(x)
            loss = self.criterion(reduction='none', **kwargs)(outputs, y)
            
            for idx in group_ids:
                losses[idx] = loss[groups==idx].mean()
        
        self.group_weights = self.group_weights * torch.exp(self.group_weights_lr*losses.data)
        self.group_weights = self.group_weights/(self.group_weights.sum())

        loss = losses @ self.group_weights

        
        return loss