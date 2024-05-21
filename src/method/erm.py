import torch.optim
from .generic_one_step_algo import GenericOneStepAlgo
from .. import mloss


class ERM(GenericOneStepAlgo):
    """Single source method which minimizes source i risk"""
    def __init__(self, device, model, loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): model
            loss_type (str): Loss name in pytorch.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        self.loss_type = loss_type

        super().__init__(device=device,
                         model=model,
                         loss=mloss.LossNLL(device, loss_type),
                         optimizer=optimizer,
                         **kwargs
                         )

    def __str__(self):
        return self.__class__.__name__
    

    
    