import torch
import numpy as np
import sys
import copy

from .erm import ERM
from .dip import DIP
from .cip import CIP
from .labelcorr import CIP_LabelCorr_CIP, CIP_LabelCorr_DIP, Src_LabelCorr_Src
from .jointdip import CIP_JointCIPDIP
from .labelcorr_jointdip import CIP_LabelCorr_JointCIPDIP
from .irm import IRM
from .vrex import VREx
from .groupdro import groupDRO
from .fish import Fish

def alg_init(algorithm, device, model, optimizer='Adam', task='classification', nb_classes=None, n_domains=1,
             alg_args=None, modelA=None, pretrained_modelA=False, **kwargs):
    
    """
    Initialize a DA algorithm.
    
    Args:
        device (str): cuda or cpu
        model (mmodel): model mapping X to Y
        optimizer (str): e.g. 'Adam'
        task (str): regression' or 'classification'
        nb_classes (int): number of classes in classification
        n_domains (int): number of domains (including target)
        alg_args (dict): dictionary of DA algorithm arguments.
        modelA (mmodel): modelA used in algorithm which needs another model during training.
        pretrained_modelA (bool): whether modelA is pretrained.
        kwargs: optimizer arguments
    """
    
    if task == 'classification':
        loss_type = 'CrossEntropyLoss'
    elif task == 'regression':
        loss_type = 'MSELoss'
    else:
        raise ValueError('Invalid Task')
    
    if pretrained_modelA:
        alg_args['lamCIP_A'] = 0.
    
    if algorithm in ['ERM', 'ERM_Pool', 'Tar']: # or 'Src', 'SrcPool'
        return ERM(device, model, loss_type, optimizer, **kwargs)
    
    elif algorithm == 'DIP_mean':
        return DIP(device, model, alg_args['lamDIP'], 'mean', nb_classes, loss_type, optimizer, **kwargs)
    
    elif algorithm == 'DIP_MMD':
        return DIP(device, model, alg_args['lamDIP'], 'MMD', nb_classes, loss_type, optimizer, **kwargs)
    
    elif algorithm == 'DIP_Pool_mean':
        return DIP(device, model, alg_args['lamDIP'], 'mean', nb_classes, loss_type, optimizer, **kwargs)

    elif algorithm == 'DIP_Pool_MMD':
        return DIP(device, model, alg_args['lamDIP'], 'MMD', nb_classes, loss_type, optimizer, **kwargs)

    elif algorithm == 'CIP_mean':
        return CIP(device, model, alg_args['lamCIP'], 'mean', nb_classes, loss_type, optimizer, **kwargs)
    
    elif algorithm == 'CIP_MMD':
        return CIP(device, model, alg_args['lamCIP'], 'MMD', nb_classes, loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-ERM':      # LabelCorr, Src_LabelCorr_Src
        return Src_LabelCorr_Src(device, model, modelA, pretrained_modelA, nb_classes, loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-CIP_mean': # CIP_LabelCorr_CIP_mean
        return CIP_LabelCorr_CIP(device, model, alg_args['lamCIP'], modelA, alg_args['lamCIP_A'],
                                 'mean', pretrained_modelA, nb_classes,
                                 loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-CIP_MMD':  # CIP_LabelCorr_CIP_MMD
        return CIP_LabelCorr_CIP(device, model, alg_args['lamCIP'], modelA, alg_args['lamCIP_A'],
                                 'MMD', pretrained_modelA, nb_classes,
                                 loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-DIP_mean': # CIP_LabelCorr_DIP_mean
        return CIP_LabelCorr_DIP(device, model, alg_args['lamDIP'], modelA, alg_args['lamCIP_A'],
                                 'mean', pretrained_modelA, nb_classes,
                                 loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-DIP_MMD':  # CIP_LabelCorr_DIP_MMD
        return CIP_LabelCorr_DIP(device, model, alg_args['lamDIP'], modelA, alg_args['lamCIP_A'],
                                 'MMD', pretrained_modelA, nb_classes,
                                 loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-DIP_Pool_mean':
        return CIP_LabelCorr_DIP(device, model, alg_args['lamDIP'], modelA, alg_args['lamCIP_A'],
                                 'mean', pretrained_modelA, nb_classes,
                                 loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-DIP_Pool_MMD': 
        return CIP_LabelCorr_DIP(device, model, alg_args['lamDIP'], modelA, alg_args['lamCIP_A'],
                                 'MMD', pretrained_modelA, nb_classes,
                                 loss_type, optimizer, **kwargs)

    elif algorithm == 'JointDIP':    # CIP_JointCIPDIP_MMD
        return CIP_JointCIPDIP(device, model, alg_args['lamDIP'], modelA, alg_args['lamCIP_A'],
                               'MMD', pretrained_modelA, nb_classes,
                               loss_type, optimizer, **kwargs)
    
    elif algorithm == 'JointDIP_Pool':  
        return CIP_JointCIPDIP(device, model, alg_args['lamDIP'], modelA, alg_args['lamCIP_A'],
                               'MMD', pretrained_modelA, nb_classes,
                               loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-JointDIP': # CIP_LabelCorr_JointCIPDIP_MMD
        return CIP_LabelCorr_JointCIPDIP(device, model, alg_args['lamDIP'], modelA, alg_args['lamCIP_A'],
                                         'MMD', pretrained_modelA, nb_classes,
                                         loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IW-JointDIP_Pool': 
        return CIP_LabelCorr_JointCIPDIP(device, model, alg_args['lamDIP'], modelA, alg_args['lamCIP_A'],
                                         'MMD', pretrained_modelA, nb_classes,
                                         loss_type, optimizer, **kwargs)
    
    elif algorithm == 'IRM':
        return IRM(device, model, alg_args['lamIRM'], alg_args['anneal_step'], loss_type, optimizer, **kwargs)
    
    elif algorithm == 'VREx':
        return VREx(device, model, alg_args['lamVREx'], alg_args['anneal_step'], loss_type, optimizer, **kwargs)
    
    elif algorithm == 'groupDRO':
        return groupDRO(device, model, alg_args['group_weights_lr'], n_domains, loss_type, optimizer, **kwargs)
    
    elif algorithm == 'Fish':
        return Fish(device, model, alg_args['meta_lr'], alg_args['meta_steps'], loss_type, optimizer, **kwargs)
    
    else:
        raise ValueError('Algorithm not implemented')
        
        
def alg_fit(alg, algorithm, dataloaders, grouper, srcId=None, tarId=None, 
            epochs=10, verbose_every=1, **kwargs):
    """
    A general function which can fit a DA algorithm
    
    Args:
        alg (BaseAlg): algorithm object used for trianing.
        algorithm (str): name of the algorithm
        dataloaders: list of dataloaders
        srcId (list of int): indices of the single source domain for methods which does not use all source domains
                             (e.g. Src, DIP, second step of CIP_LabelCorr_CIP/DIP, CIP_JointCIPDIP)
        tarId (int): index of target domain. Defaults to the last one.
        epochs (int): number of epochs used for training
        verbose_every (int): use tqdm / check accuracy every verbose_every epoch.
        kwargs (optional): {additional_dataloaders: dataloaders used to check accuracy during training 
                                                    (every verbose_every epoch)
                            additional_split_names: corresponding name of additional_dataloaders}
    """
    
    if tarId is None:
        tarId = len(dataloaders)-1
    source_dataloaders = [dataloaders[i] for i in range(len(dataloaders)) if i != tarId]
    srcIds = [i for i in range(len(dataloaders)) if i != tarId]
    
    
    if algorithm == 'ERM':           # Src
        return alg.fit([dataloaders[i] for i in srcId], grouper, None, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'ERM_Pool':    # SrcPool
        return alg.fit(source_dataloaders, grouper, None, epochs, verbose_every, **kwargs)
        
    elif algorithm == 'Tar':
        return alg.fit([dataloaders[tarId]], grouper, None, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'DIP_mean':
        return alg.fit([dataloaders[i] for i in srcId]+[dataloaders[tarId]], grouper, -1, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'DIP_MMD':
        return alg.fit([dataloaders[i] for i in srcId]+[dataloaders[tarId]], grouper, -1, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'DIP_Pool_mean':
        return alg.fit([dataloaders[i] for i in srcIds]+[dataloaders[tarId]], grouper, -1, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'DIP_Pool_MMD':
        return alg.fit([dataloaders[i] for i in srcIds]+[dataloaders[tarId]], grouper, -1, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'CIP_mean':
        return alg.fit(source_dataloaders, grouper, None, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'CIP_MMD':
        return alg.fit(source_dataloaders, grouper, None, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-ERM':       # LabelCorr, Src_LabelCorr_Src
        return alg.fit(dataloaders, grouper, srcIds, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-CIP_mean':  # CIP_LabelCorr_CIP_mean
        return alg.fit(dataloaders, grouper, srcIds, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-CIP_MMD':   # CIP_LabelCorr_CIP_MMD
        return alg.fit(dataloaders, grouper, srcIds, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-DIP_mean':  # CIP_LabelCorr_DIP_mean
        return alg.fit(dataloaders, grouper, srcId, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-DIP_MMD':   # CIP_LabelCorr_DIP_MMD
        return alg.fit(dataloaders, grouper, srcId, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-DIP_Pool_mean':  # CIP_LabelCorr_DIP_mean
        return alg.fit(dataloaders, grouper, srcIds, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-DIP_Pool_MMD':   # CIP_LabelCorr_DIP_MMD
        return alg.fit(dataloaders, grouper, srcIds, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'JointDIP':     # CIP_JointCIPDIP_MMD
        return alg.fit(dataloaders, grouper, srcId, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'JointDIP_Pool':   
        return alg.fit(dataloaders, grouper, srcIds, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-JointDIP':  # CIP_LabelCorr_JointCIPDIP_MMD
        return alg.fit(dataloaders, grouper, srcId, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IW-JointDIP_Pool':  
        return alg.fit(dataloaders, grouper, srcIds, tarId, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'IRM':
        return alg.fit(source_dataloaders, grouper, None, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'VREx':
        return alg.fit(source_dataloaders, grouper, None, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'groupDRO':
        return alg.fit(source_dataloaders, grouper, None, epochs, verbose_every, **kwargs)
    
    elif algorithm == 'Fish':
        return alg.fit(source_dataloaders, grouper, None, epochs, verbose_every, **kwargs)
    
    else:
        raise ValueError('Algorithm not implemented')