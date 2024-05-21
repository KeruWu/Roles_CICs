import torch
import torch.nn as nn
from . import helpers
from .base_loss import BaseLoss

class LossJointDIP(BaseLoss):
    """
    Loss + Joint Domain Invariant Penalty (JointDIP). JointDIP is default to MMD matching.
    """
    def __init__(self, device, nb_classes, lamDIP, discrepType='MMD', loss_type='CrossEntropyLoss', freeze_bn=False, groupdip=False):
        super().__init__(device)
        self.lamDIP = lamDIP
        self.nb_classes = nb_classes
        self.discrepType = discrepType
        self.freeze_bn = freeze_bn
        self.groupdip = groupdip
        
        if self.discrepType == 'MMD':
            self.discrepancy = helpers._rbf_mmd2
        else: # self.discrepType = 'mean':
            self.discrepancy = helpers._diff_mean
        
        self.criterion = getattr(nn, loss_type)
    
    
    def dip_penalty(self, source_feats, source_feats_joint,
                          target_feats, target_feats_joint, source_y=None, **kwargs):
        
        source_jointed = torch.cat([source_feats, source_feats_joint], axis=1)
        target_jointed = torch.cat([target_feats, target_feats_joint], axis=1)
        
        if 'weight' not in kwargs:
            penalty = self.discrepancy(source_jointed, target_jointed)
        else:
            # need to weigh the data points in the DIP penalty
            source_y = source_y.to(self.device)
            labels_onehot = torch.zeros((source_y.shape[0], self.nb_classes), dtype=torch.float).to(self.device)
            labels_onehot = labels_onehot.scatter_(1, source_y.reshape(-1, 1), 1)
            sample_weight = labels_onehot.matmul(kwargs['weight'])
            penalty = self.discrepancy(source_jointed, target_jointed, sample_weight=sample_weight)

        return penalty
    
    
    def __call__(self, data, model, groups=None, featurizer_joint=None, **kwargs):
        
        
        
        if groups is None:
            target_x = data[-1][0].to(self.device)
            target_feats = model.featurizer(target_x)
            target_feats_joint = featurizer_joint(target_x)
            
            nb_envs = len(data)-1
            loss = 0.
            penalty = 0.
            
            for i in range(nb_envs):
                source_x, source_y = data[i][0].to(self.device), data[i][1].to(self.device)
                source_feats = model.featurizer(source_x)
                source_feats_joint = featurizer_joint(source_x)
                source_outputs = model.classifier(source_feats)
                loss += self.criterion(**kwargs)(source_outputs, source_y)
                penalty += self.dip_penalty(source_feats, source_feats_joint,
                                            target_feats, target_feats_joint,
                                            source_y, **kwargs)
            
            loss_penalty = loss + self.lamDIP * penalty
            return loss_penalty / nb_envs
        
        else:
            source_x, source_y = data[0][0].to(self.device), data[0][1].to(self.device)
            target_x = data[-1][0].to(self.device)
            
            if self.freeze_bn:
                
                source_feats = model.featurizer(source_x)
                helpers.freeze_bn_layers(model)
                target_feats = model.featurizer(target_x)
                helpers.unfreeze_bn_layers(model)
                
                source_feats_joint = featurizer_joint(source_x)
                target_feats_joint = featurizer_joint(target_x)
            
            else:
                batch_size = source_x.shape[0]
                
                x = torch.cat([source_x, target_x], dim=0)
                feats = model.featurizer(x)
                source_feats, target_feats = feats[:batch_size], feats[batch_size:]
                feats_joint = featurizer_joint(x)
                source_feats_joint, target_feats_joint = feats_joint[:batch_size], feats_joint[batch_size:]
            
            source_outputs = model.classifier(source_feats)
            
            loss = self.criterion(**kwargs)(source_outputs, source_y)
            
            if self.groupdip:
                penalty = 0.
                group_ids = torch.unique(groups)
                nb_envs = len(group_ids)
                
                for id1 in group_ids:
                    penalty += self.dip_penalty(source_feats[groups==id1], source_feats_joint[groups==id1],
                                                target_feats, target_feats_joint, source_y[groups==id1], **kwargs)
                penalty /= nb_envs
                
            else:
                penalty = self.dip_penalty(source_feats, source_feats_joint,
                                           target_feats, target_feats_joint,
                                           source_y, **kwargs)
            
            
            loss_penalty = loss + self.lamDIP * penalty
            return loss_penalty
        

#     def __call__(self, data, model, dataInds = None, class_weight=None, **kwargs):
#         # kwargs must contain modelCIP
#         # the first two indexes of dataInds will be used for DIP
#         sourceInd = dataInds[0]
#         targetInd = dataInds[1]

#         inputs, labels = data[sourceInd][0].to(self.device), data[sourceInd][1].to(self.device)
#         inputs_t = data[targetInd][0].to(self.device)

#         outputs = torch.cat([model.forward2feat(inputs),
#                              kwargs['modelCIP'].forward2feat(inputs)], axis=1)
#         outputs_t = torch.cat([model.forward2feat(inputs_t),
#                                kwargs['modelCIP'].forward2feat(inputs_t)], axis=1)

#         if self.discrepType == 'MMD':
#             discrepancy = helpers._rbf_mmd2
#         else: # self.discrepType = 'mean':
#             discrepancy = helpers._diff_mean

#         if class_weight is None:
#             self.penalty = self.lamDIP * discrepancy(outputs, outputs_t)
#         else:
#             # need to weigh the data points in the DIP penalty
#             labels_onehot = torch.zeros((labels.shape[0], self.nb_classes), dtype=torch.float).to(self.device)
#             labels_onehot = labels_onehot.scatter_(1, labels.reshape(-1, 1), 1)
#             sample_weight = labels_onehot.matmul(class_weight)
#             self.penalty = self.lamDIP * discrepancy(outputs, outputs_t, sample_weight=sample_weight)

#         return self.penalty