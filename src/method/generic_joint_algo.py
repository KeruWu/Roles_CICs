"""
Generic two step DA method 
which does
1. train first model (like CIP)
2. use both features from the first model and the final model to do matching
(like DIP)
"""
import torch.optim as optim
import copy
from .base_algo import BaseAlgo
from .mtraining import Trainer

class GenericJointAlgo(BaseAlgo):
    """Generic two step method
    the first step is learning the conditional invariant feature
    the second step is jointly matching CIP/DIP features
    """
    def __init__(self, device, model, loss, optimizer, modelA=None, lossA=None, optimizerA=None,
                 pretrained_modelA=False, nb_classes=None, **kwargs):
        """Initialization with all needs

        Args:
            device (str): cuda or cpu.
            model (mmodel): the model for training
            loss (mloss): loss function, e.g. mloss.LossNLL(device)
            optimizer (str): optimizer used for model.
            modelA (torch.model with featurizer): the model used for joint matching. If None, 
            optimizerA (str): optimizer used for modelA. If None, same as optimizer
                              If None, modelA is initialized from a copy of model.
            pretrained_modelA (bool): whether modelA is pretrained. If not, train modelA first.
            kwargs: optimizer parameters, e.g. lr, weight_decay.
        """
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), **kwargs)
        
        self.modelA = copy.deepcopy(self.model) if modelA is None else modelA
        self.lossA = loss if lossA is None else lossA
        optimizerA = optimizer if optimizerA is None else optimizerA
        self.optimizerA = getattr(optim, optimizerA)(self.modelA.parameters(), **kwargs)

        self.trainerA = Trainer(self.modelA, self.lossA, self.optimizerA, self.device)
        self.trainer = Trainer(self.model, self.loss, self.optimizer, self.device)
        
        self.pretrained_modelA = pretrained_modelA
        self.nb_classes = nb_classes
        
    def fit(self, dataloaders, grouper=None, srcIds=None, tarId=None, epochs=10, verbose_every=1, **kwargs):
        """Fit the model

        Args:
            dataloaders: list of dataloaders. All source dataloaders are used for pretraining modelA by default.
            grouper: if None, each dataloader corresponds to each domain,
                     otherwise, grouper.metadata_to_group gives the group id (specially designed for WILDS).
            srcIds (list of int): Indices of source dataloaders used for Joint matching.
            tarId (int): which index of dataloaders corresponds to the target domain. 
                         If None: target domain is the last dataloader.
            epochs (int): number of epochs for training.
            verbose_every (int): use tqdm / check accuracy every verbose_every epoch.
            kwargs (optional): {additional_dataloaders: dataloaders used to check accuracy during training 
                                                        (every verbose_every epoch)
                                additional_split_names: corresponding name of additional_dataloaders}
        """
        dataloaders1 = [dataloaders[i] for i in range(len(dataloaders)) if i != tarId]

        if not self.pretrained_modelA:
            self.trainerA.train(dataloaders1, grouper, None, epochs, verbose_every, **kwargs)
        
        self.modelA = self.modelA.eval()
        for param in self.modelA.parameters():
            param.requires_grad = False

        dataloaders2 = [dataloaders[i] for i in srcIds]
        dataloaders2.append(dataloaders[tarId])

        return self.trainer.train(dataloaders2, grouper, None, epochs, verbose_every, 
                                  featurizer_joint=self.modelA.featurizer, **kwargs)
        
