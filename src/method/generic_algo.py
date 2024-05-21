"""
Generic one step DA method 
which takes a loss and a penalty, 
and minimizes the loss + penalty on a subset of dataloaders
"""

from .base_algo import BaseAlgo
from .mtraining import Trainer
import torch.optim as optim

class GenericAlgo(BaseAlgo):
    """Generic algorithm"""
    def __init__(self, device, model, loss, optimizer, **kwargs):
        """Initialization with all needs

        Args:
            device (str): cuda or cpu
            model (mmodel): model mapping X to Y
            loss (mloss): loss function, e.g. mloss.LossNLL(device)
            optimizer (str): e.g. 'Adam'
            kwargs: optimizer arguments
        """
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = getattr(optim, optimizer)
        self.optimizer_args_dict = kwargs
        self.trainer = Trainer(self.model, self.loss, self.optimizer, self.device, alg=self)
        

    def fit(self, dataloaders, grouper=None, tarId=None, epochs=10, verbose_every=1, **kwargs):

        return self.trainer.train(dataloaders, grouper, tarId, epochs, verbose_every, **kwargs)
    
    def process_batch(self, data, groups=None, **kwargs):
        pass
    