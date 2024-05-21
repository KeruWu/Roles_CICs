import torch
import torch.nn as nn
import torch.autograd as autograd
from .base_loss import BaseLoss


class LossIRM(BaseLoss):
    """
    Loss + The invariant risk minimization (IRM) penalty
    """
    def __init__(self, device, lamIRM, anneal_step, loss_type='CrossEntropyLoss'):
        super().__init__(device)
        self.lamIRM = lamIRM
        self.anneal_penalty = 1. if lamIRM >= 1 else 0.
        self.step = 0
        self.anneal_step = anneal_step
        self.criterion = getattr(nn, loss_type)
        
    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        
        # another way to calculate irm penalty:
        # g = autograd.grad(loss.mean(), self.scale, create_graph=True)[0]
        # self.penalty += self.anneal_penalty * torch.sum(g**2)
        
        return result
        
    def __call__(self, data, model, groups=None, **kwargs):
        
        # lambda annealing: 
        # Before anneal_step steps, use lambda=1 if lamIRM>=1 and lambda=0 if lamIRM<1.
        # After anneal_strep steps, use lambda=lamIRM.
        
        if self.step == self.anneal_step:
            self.anneal_penalty = self.lamIRM
        self.step += 1
        
        loss = 0.
        penalty = 0.
        
        if groups is None:
            nb_envs = len(data)
            for i in range(nb_envs):
                x, y = data[i][0].to(self.device), data[i][1].to(self.device)
                outputs = model(x)
                loss += self.criterion(reduction='mean')(outputs, y)

                self.scale = torch.tensor(1.).to(self.device).requires_grad_()
                losses = self.criterion(reduction='none')(outputs * self.scale, y)
                penalty += self.irm_penalty(losses)
                
            loss_penalty = loss + penalty * self.anneal_penalty
        
            return loss_penalty / nb_envs
                
        else:
            group_ids = torch.unique(groups)
            
            x, y = data[0][0].to(self.device), data[0][1].to(self.device)
            outputs = model(x)
            loss += self.criterion(reduction='mean', **kwargs)(outputs, y)
            
            for idx in group_ids:
                self.scale = torch.tensor(1.).to(self.device).requires_grad_()
                losses = self.criterion(reduction='none', **kwargs)(outputs[group_ids==idx]*self.scale, y[group_ids==idx])
                penalty += self.irm_penalty(losses)

            loss_penalty = loss + penalty * self.anneal_penalty
        
            return loss_penalty