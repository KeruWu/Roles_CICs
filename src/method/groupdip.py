import torch.optim
from .generic_one_step_algo import GenericOneStepAlgo
from .. import mloss 


class groupDIP(GenericOneStepAlgo):
    """Single source domain invariant penalty (DIP) method
    which minimizes source risk + 
    penalty on one source and target covariates"""
    def __init__(self, device, model, lamDIP=0.1, discrepType='mean', nb_classes=None, 
                loss_type='CrossEntropyLoss', optimizer='Adam', freeze_bn=False, **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): model
            lamDIP (float): regularization param of DIP penalty
            discrepType (str): type of discrepancy, 'mean' or 'MMD'
            nb_classes (int): number of classes.
            loss_type (str): Loss name in pytorch.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            freeze_bn (bool): freeze batchnorm layers when calling model(target_x) or not.
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        self.lamDIP = lamDIP
        self.discrepType = discrepType
        self.loss_type = loss_type
        self.freeze_bn = freeze_bn

        super().__init__(device=device,
                         model=model,
                         loss=mloss.LossgroupDIP(device, nb_classes, lamDIP, discrepType, loss_type, freeze_bn),
                         optimizer=optimizer,
                         **kwargs
                         )

    def __str__(self):
        return f"{self.__class__.__name__}_{self.discrepType}_lamDIP={self.lamDIP}"