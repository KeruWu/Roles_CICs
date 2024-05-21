import torch.optim
from .generic_one_step_algo import GenericOneStepAlgo
from .. import mloss


class IRM(GenericOneStepAlgo):
    """
    Invariant Risk Minimization (IRM) method
    """
    def __init__(self, device, model, lamIRM=1., anneal_step=0, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): model
            lamIRM (float): regularization param of IRM penalty
            anneal_step (int): number of iterations for lambda annealing.
            loss_type (str): Loss name in pytorch.
            optimizer (str): optimization. Defaults to 'Adam'.
            kwargs: optimizer arguments
        """

        self.lamIRM = lamIRM
        self.anneal_step = anneal_step
        self.loss_type = loss_type

        super().__init__(device=device,
                         model=model,
                         loss=mloss.LossIRM(device, lamIRM, anneal_step, loss_type),
                         optimizer=optimizer,
                         **kwargs
                         )

    def __str__(self):
        return f'{self.__class__.__name__}_lamIRM={self.lamIRM}_anneal_step={self.anneal_step}'