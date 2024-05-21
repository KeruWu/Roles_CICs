import torch.optim
from .generic_one_step_algo import GenericOneStepAlgo
from .. import mloss


class CIP(GenericOneStepAlgo):
    """
    Conditional invariant penalty (CIP) method
    which minimizes pooled source risk + 
    penalty on conditional invariance X|Y
    """
    def __init__(self, device, model, lamCIP=0.1, discrepType='mean', nb_classes=None, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): model
            lamCIP (float): regularization param of CIP penalty
            discrepType (str): type of discrepancy, 'mean' or 'MMD'
            nb_classes (int): number of classes.
            loss_type (str): Loss name in pytorch.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        self.lamCIP = lamCIP
        self.discrepType = discrepType
        self.loss_type = loss_type

        super().__init__(device=device,
                         model=model,
                         loss=mloss.LossCIP(device, nb_classes, lamCIP, discrepType, loss_type),
                         optimizer=optimizer,
                         **kwargs
                         )

    def __str__(self):
        return f"{self.__class__.__name__}_{self.discrepType}_lamCIP={self.lamCIP}"