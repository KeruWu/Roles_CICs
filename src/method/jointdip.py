import torch
from .generic_joint_algo import GenericJointAlgo
from .. import mloss

class CIP_JointCIPDIP(GenericJointAlgo):
    """First estimate CIP as invariant features Second, do joint matching with DIP not much guarantees in general but can alleviate label-flipping
    """
    def __init__(self, device, model, lamDIP=0.1, modelA=None, lamCIP_A=None, 
                 discrepType='MMD', pretrained_modelA=False, nb_classes=None, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', freeze_bn=False, groupdip=False,
                 **kwargs):
        """default initialization

        Args:
            device (device): cuda or cpu
            model (mmodel): model
            lamDIP (float): regularization param of JointDIP penalty
            modelA (mmodel): model with featurizer for joint matching
            lamCIP_A (float): regularization param of CIP penalty
            discrepType (str): type of discrepancy, 'mean' or 'MMD'
            pretrained_modelA (bool): whether modelA is pretrained or not.
            nb_classes (int): number of classes.
            loss_type (str): Loss name in pytorch.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            freeze_bn (bool): freeze batchnorm layers when calling model(target_x) or not.
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        
        self.lamCIP_A = lamCIP_A
        self.lamDIP = lamDIP
        self.discrepType = discrepType
        self.freeze_bn = freeze_bn
        self.groupdip = groupdip

        super().__init__(device=device, 
                         model=model, 
                         loss=mloss.LossJointDIP(device, nb_classes, lamDIP, discrepType, loss_type, freeze_bn, groupdip),
                         optimizer=optimizer,
                         modelA=modelA,
                         lossA=mloss.LossCIP(device, nb_classes, lamCIP_A, discrepType, loss_type),
                         optimizerA=optimizer,
                         pretrained_modelA=pretrained_modelA,
                         nb_classes=nb_classes,
                         **kwargs)
            
    def __str__(self):
        return f'{self.__class__.__name__}_{self.discrepType}'