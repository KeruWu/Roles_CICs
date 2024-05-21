import torch.optim
from .generic_label_corr_algo import GenericLabelCorrAlgo
from .. import mloss


class CIP_LabelCorr_CIP(GenericLabelCorrAlgo):
    """
    Conditional invariant penalty (CIP) method
    to do label shift correction, then followed by CIP for prediction
    """
    def __init__(self, device, model, lamCIP=0.1, modelA=None, lamCIP_A=None, 
                 discrepType='mean', pretrained_modelA=False, nb_classes=None, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): second model used for importrance-weighted CIP
            lamCIP (float): CIP penalty for the second importance-weighted CIP
            modelA (mmodel): first model used for getting CIP features and importance weights
            lamCIP_A (float): CIP penalty for the first CIP
            discrepType (str): type of discrepancy, 'mean' or 'MMD'
            pretrained_modelA (bool): whether modelA is pretrained or not.
            nb_classes (int): number of classes.
            loss_type (str): Loss name in pytorch.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        self.lamCIP_A = lamCIP_A
        self.lamCIP = lamCIP
        self.discrepType = discrepType
        self.loss_type = loss_type
        self.require_target = False

        super().__init__(device=device, 
                         model=model, 
                         loss=mloss.LossCIP(device, nb_classes, lamCIP, discrepType, loss_type),
                         optimizer=optimizer,
                         modelA=modelA,
                         lossA=mloss.LossCIP(device, nb_classes, lamCIP_A, discrepType, loss_type),
                         optimizerA=optimizer,
                         pretrained_modelA=pretrained_modelA,
                         nb_classes=nb_classes,
                         **kwargs)

    def __str__(self):
        return f'{self.__class__.__name__}_{self.discrepType}'
    


class CIP_LabelCorr_DIP(GenericLabelCorrAlgo):
    """
    Conditional invariant penalty (CIP) method
    to do label shift correction, then followed by DIP for prediction
    No guarantees in general
    """
    def __init__(self, device, model, lamDIP=0.1, modelA=None, lamCIP_A=None, 
                 discrepType='mean', pretrained_modelA=False, nb_classes=None, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): second model used for importrance-weighted DIP
            lamDIP (float): DIP penalty for the second importance-weighted DIP
            modelA (mmodel): first model used for getting CIP features and importance weights
            lamCIP_A (float): CIP penalty for the first CIP
            discrepType (str): type of discrepancy, 'mean' or 'MMD'
            pretrained_modelA (bool): whether modelA is pretrained or not.
            nb_classes (int): number of classes.
            loss_type (str): Loss name in pytorch.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        self.lamCIP_A = lamCIP_A
        self.lamDIP = lamDIP
        self.discrepType = discrepType
        self.loss_type = loss_type
        self.require_target = True

        super().__init__(device=device, 
                         model=model, 
                         loss=mloss.LossDIP(device, nb_classes, lamDIP, discrepType, loss_type),
                         optimizer=optimizer,
                         modelA=modelA,
                         lossA=mloss.LossCIP(device, nb_classes, lamCIP_A, discrepType, loss_type),
                         optimizerA=optimizer,
                         pretrained_modelA=pretrained_modelA,
                         nb_classes=nb_classes,
                         **kwargs)

    def __str__(self):
        return f'{self.__class__.__name__}_{self.discrepType}'
  
  
    
    
class Src_LabelCorr_Src(GenericLabelCorrAlgo):
    """
    SrcPool method to do label correction, followed by SrcPool
    no guarantees in general
    """
    def __init__(self, device, model, modelA=None, 
                 pretrained_modelA=False, nb_classes=None, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): second model used for importrance-weighted ERM
            modelA (mmodel): first model used for getting importance weights
            pretrained_modelA (bool): whether modelA is pretrained or not.
            nb_classes (int): number of classes.
            loss_type (str): Loss name in pytorch.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        self.loss_type = loss_type
        self.require_target = False

        super().__init__(device=device, 
                         model=model, 
                         loss=mloss.LossNLL(device, loss_type),
                         optimizer=optimizer,
                         modelA=modelA,
                         lossA=mloss.LossNLL(device, loss_type),
                         optimizerA=optimizer,
                         pretrained_modelA=pretrained_modelA,
                         nb_classes=nb_classes,
                         **kwargs)

    def __str__(self):
        return self.__class__.__name__