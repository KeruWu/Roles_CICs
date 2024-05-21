"""
Generic one step DA method 
which takes a loss and a penalty, 
and minimizes the loss + penalty on a subset of dataloaders
"""

import torch.optim as optim
from .base_algo import BaseAlgo
from .mtraining import Trainer


class GenericOneStepAlgo(BaseAlgo):
    """Generic one step DA algorithm which minimizes the loss + penalty"""
    def __init__(self, device, model, loss, optimizer, **kwargs):
        """Initialization with all needs

        Args:
            device (str): cuda or cpu
            model (mmodel): model mapping X to Y
            loss (mloss): loss function, e.g. mloss.LossNLL(device)
            optimizer (str): e.g. 'Adam'
            kwargs: optimizer parameters, e.g. lr, weight_decay.
        """
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), **kwargs)
        self.trainer = Trainer(self.model, self.loss, self.optimizer, self.device)

    def fit(self, dataloaders, grouper=None, tarId=None, epochs=10, verbose_every=1, **kwargs):
        """Fit the model

        Args:
            dataloaders: list of dataloaders
            grouper: if None, each dataloader corresponds to each domain,
                     otherwise, grouper.metadata_to_group gives the group id (specially designed for WILDS).
            tarId (int): which index of dataloaders corresponds to the target domain. 
                         If None: target domain is the last dataloader.
            epochs (int): number of epochs for training.
            verbose_every (int): use tqdm / check accuracy every verbose_every epoch.
            kwargs (optional): {additional_dataloaders: dataloaders used to check accuracy during training 
                                                        (every verbose_every epoch)
                                additional_split_names: corresponding name of additional_dataloaders}
        """
        return self.trainer.train(dataloaders, grouper, tarId, epochs, verbose_every, **kwargs)
