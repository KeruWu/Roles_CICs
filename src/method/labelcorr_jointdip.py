import torch.optim
from .generic_label_corr_joint_algo import GenericLabelCorrJointAlgo
from .. import mloss


class CIP_LabelCorr_JointCIPDIP(GenericLabelCorrJointAlgo):
    """Conditional invariant penalty (CIP) method
    to do label shift correction, then followed by JointCIPDIP for prediction"""
    def __init__(self, device, model, lamDIP=0.1, modelA=None, lamCIP_A=None, 
                 discrepType='MMD', pretrained_modelA=False, nb_classes=None, 
                 loss_type='CrossEntropyLoss', optimizer='Adam', **kwargs):
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
            kwargs (dict): optimizer arguments, e.g. lr, weight_decay
        """
        self.lamCIP_A = lamCIP_A
        self.lamDIP = lamDIP
        self.discrepType = discrepType

        super().__init__(device=device, 
                         model=model, 
                         loss=mloss.LossJointDIP(device, nb_classes, lamDIP, discrepType, loss_type),
                         optimizer=optimizer,
                         modelA=modelA,
                         lossA=mloss.LossCIP(device, nb_classes, lamCIP_A, discrepType, loss_type),
                         optimizerA=optimizer,
                         pretrained_modelA=pretrained_modelA,
                         nb_classes=nb_classes,
                         **kwargs)

    def __str__(self):
        return f'{self.__class__.__name__}_{self.discrepType}'
  