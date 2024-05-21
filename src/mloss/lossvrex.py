import torch
import torch.nn as nn
from .base_loss import BaseLoss


class LossVREx(BaseLoss):
    """
    Loss + The Variance of training risks (VREx) penalty
    """
    def __init__(self, device, lamVREx, anneal_step, loss_type='CrossEntropyLoss'):
        super().__init__(device)
        self.lamVREx = lamVREx
        self.anneal_penalty = 1. if lamVREx >= 1 else 0.
        self.step = 0
        self.anneal_step = anneal_step
        self.criterion = getattr(nn, loss_type)

    def __call__(self, data, model, groups=None, **kwargs):
    
        # lambda annealing: 
        # Before anneal_step steps, use lambda=1 if lamVREx>=1 and lambda=0 if lamVREx<1.
        # After anneal_strep steps, use lambda=lamVREx.
        
        if self.step == self.anneal_step:
            self.anneal_penalty = self.lamVREx
        self.step += 1
        
        if groups is None:
            losses = torch.zeros(len(data)).to(self.device)
            
            for i in range(len(data)):
                x, y = data[i][0].to(self.device), data[i][1].to(self.device)
                outputs = model(x)
                losses[i] = self.criterion(reduction='mean', **kwargs)(outputs, y)

        else:
            group_ids = torch.unique(groups)
            nb_envs = len(group_ids)
            losses = torch.zeros(len(data)).to(self.device)
            
            x, y = data[0][0].to(self.device), data[0][1].to(self.device)
            outputs = model(x)
            loss = self.criterion(reduction='none', **kwargs)(outputs, y)
            
            for i, idx in enumerate(group_ids):
                losses[i] = loss[group_ids==idx].mean()

        loss_mean = losses.mean()
        penalty = ((losses - loss_mean) ** 2).mean()

        loss_penalty = loss_mean + penalty * self.anneal_penalty
        return loss_penalty