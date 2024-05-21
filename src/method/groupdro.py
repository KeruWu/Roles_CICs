import torch.optim
from .generic_one_step_algo import GenericOneStepAlgo
from .. import mloss


class groupDRO(GenericOneStepAlgo):
    """
    group distributionally robust optimization (groupDRO) method
    which minimizes pooled source risks weighted by group weights
    """
    def __init__(self, device, model, group_weights_lr=1e-2, n_domains=None, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): model
            group_weights_lr (float): learning rate of group weights
            n_domains (int): number of domains
            loss_type (str): Loss name in pytorch.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        self.group_weights_lr = group_weights_lr
        self.loss_type = loss_type

        super().__init__(device=device,
                         model=model,
                         loss=mloss.LossgroupDRO(device, group_weights_lr, n_domains, loss_type),
                         optimizer=optimizer,
                         **kwargs
                         )

    def __str__(self):
        return f'{self.__class__.__name__}_group_lr={self.group_weights_lr}'