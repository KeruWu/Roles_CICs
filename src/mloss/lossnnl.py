from .base_loss import BaseLoss
import torch.nn as nn


class LossNLL(BaseLoss):
    """The default negative log-likelihood loss
    """
    def __init__(self, device, loss_type='CrossEntropyLoss'):
        super().__init__(device)
        self.loss_type = loss_type
        self.criterion = getattr(nn, loss_type)

    def __call__(self, data, model, groups=None, **kwargs):
        
        if groups is None:
            nb_envs = len(data)
            loss = 0.
            for i in range(nb_envs):

                x, y = data[i][0].to(self.device), data[i][1].to(self.device)
                outputs = model(x)
                loss += self.criterion(**kwargs)(outputs, y)
            
            return loss / nb_envs
        
        else:
            x, y = data[0][0].to(self.device), data[0][1].to(self.device)
            outputs = model(x)
            loss = self.criterion(**kwargs)(outputs, y)

            return loss