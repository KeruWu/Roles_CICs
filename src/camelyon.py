## Run experiments on Camelyon17 from WILDS (https://github.com/p-lambda/wilds)

import argparse
import sys
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

sys.path.append("..")
from src import mmodel
from src import mdata
from src import method
from src.config import dataset_defaults

torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs='+', type=int, default=[0], help="Seeds in experiments")
    parser.add_argument('--algorithm', type=str, default=None, help="Algorithm(s) to run")
    parser.add_argument('--save_path', type=str, default='../results', help="Path to save results")
    parser.add_argument("--download", type=str, default=False)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--save_pred_path", type=str, default='../results', 
                        help="Path to save model predictions. If None, does not save predictions")
    
    myargs = parser.parse_args()

    
    algorithm = myargs.algorithm
    modelA = None
    pretrained_modelA = False
    
    config = dataset_defaults['Camelyon17']
    
    
    for seed in myargs.seeds:
        
        print("=================================================")
        print(f'seed = {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        
        train_loader, train_loader_grouped, target_loader, target_loader_labeled, id_val_loader, val_loader, test_loader, grouper = mdata.wilds.load_camelyon(root_dir=myargs.data_root, download=myargs.download, random_augment=True if algorithm != 'CIP' else False)
        
        additional_dataloaders = [id_val_loader, val_loader, test_loader]
        additional_split_names = ['id_val', 'val', 'test']

        model = mmodel.DenseNet(nb_classes=2).to(device)
            
        other_kwargs = {"additional_dataloaders": additional_dataloaders, 
                        "additional_split_names": additional_split_names}
        if myargs.save_pred_path is not None:
            other_kwargs['save_pred_path'] = myargs.save_pred_path
            other_kwargs['seed'] = seed
            
        t0 = time.time()


        print(algorithm)
        torch.manual_seed(seed)
        np.random.seed(seed)
      
        if algorithm == 'JointDIP-Pool': # CIP_JointCIPDIP_MMD_group
            
            lamDIP = config[algorithm]['lamDIP']
            lamCIP_A = config[algorithm]['lamCIP_A']
            if os.path.exists(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_lamCIP{lamCIP_A}_lamDIP{lamDIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy"):
                continue
            alg = method.CIP_JointCIPDIP(device, model, lamDIP=lamDIP, modelA=None, lamCIP_A=lamCIP_A,
                                         discrepType='MMD', pretrained_modelA=False, nb_classes=2, loss_type='CrossEntropyLoss', 
                                         optimizer='SGD', freeze_bn=0, lr=1e-3, weight_decay=0.01, momentum=0.9, groupdip=True)


            result1 = alg.trainerA.train([train_loader_grouped], grouper, None, epochs=10, verbose_every=1)

            alg.modelA = alg.modelA.eval()
            for param in alg.modelA.parameters():
                param.requires_grad = False
                
            print("============== First CIP finished ================")

            result2 = alg.trainer.train([train_loader_grouped, target_loader], grouper, None, epochs=10, verbose_every=1, 
                                        featurizer_joint=alg.modelA.featurizer, **other_kwargs)

            #result = [result1, result2]
            result = result2
            np.save(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_lamCIP{lamCIP_A}_lamDIP{lamDIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy", result)
            
            
            
        elif algorithm == 'JointDIP-Pool_target_labeled': # 'CIP_JointCIPDIP_MMD_group_target_labeled'
            
            lamDIP = config[algorithm]['lamDIP']
            lamCIP_A = config[algorithm]['lamCIP_A']
            if os.path.exists(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_lamCIP{lamCIP_A}_lamDIP{lamDIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy"):
                continue
            
            alg = method.CIP_JointCIPDIP(device, model, lamDIP=lamDIP, modelA=None, lamCIP_A=lamCIP_A,
                                         discrepType='MMD', pretrained_modelA=False, nb_classes=2, loss_type='CrossEntropyLoss', 
                                         optimizer='SGD', freeze_bn=0, lr=1e-3, weight_decay=0.01, momentum=0.9, groupdip=True)


            result1 = alg.trainerA.train([train_loader_grouped], grouper, None, epochs=10, verbose_every=1)

            alg.modelA = alg.modelA.eval()
            for param in alg.modelA.parameters():
                param.requires_grad = False
                
            print("============== First CIP finished ================")

            result2 = alg.trainer.train([train_loader_grouped, target_loader_labeled], grouper, None, epochs=10, verbose_every=1, 
                                        featurizer_joint=alg.modelA.featurizer, **other_kwargs)

            #result = [result1, result2]
            result = result2
            np.save(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_lamCIP{lamCIP_A}_lamDIP{lamDIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy", result)
        
        
        
        elif algorithm == 'DIP-Pool':  # groupDIP_MMD_grouped
            
            lamDIP = config[algorithm]['lamDIP']
            if os.path.exists(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_{lamDIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy"):
                continue
            alg = method.groupDIP(device, model, lamDIP=lamDIP, 
                             discrepType='MMD', nb_classes=2, loss_type='CrossEntropyLoss', 
                             optimizer='SGD', freeze_bn=0, lr=1e-3, weight_decay=0.01, momentum=0.9)
            
            result = alg.trainer.train([train_loader_grouped, target_loader], grouper, None, epochs=10, verbose_every=1, 
                                       **other_kwargs)
            
            np.save(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_{lamDIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy", result)
            
            
            
        elif algorithm == 'DIP-Pool_target_labeled':  # groupDIP_MMD_grouped_target_labeled
            
            lamDIP = config[algorithm]['lamDIP']
            if os.path.exists(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_{lamDIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy"):
                continue
            alg = method.groupDIP(device, model, lamDIP=lamDIP, 
                             discrepType='MMD', nb_classes=2, loss_type='CrossEntropyLoss', 
                             optimizer='SGD', freeze_bn=0, lr=1e-3, weight_decay=0.01, momentum=0.9)
            
            result = alg.trainer.train([train_loader_grouped, target_loader_labeled], grouper, None, epochs=10, verbose_every=1, 
                                       **other_kwargs)
            
            np.save(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_{lamDIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy", result)
          
        
        
        elif algorithm == 'CIP': # CIP_MMD_grouped
            
            lamCIP = config[algorithm]['lamCIP']
            if os.path.exists(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_{lamCIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy"):
                continue
            alg = method.CIP(device, model, lamCIP=lamCIP, 
                             discrepType='MMD', nb_classes=2, loss_type='CrossEntropyLoss', 
                             optimizer='SGD', lr=1e-3, weight_decay=0.01, momentum=0.9)
            
            result = alg.trainer.train([train_loader_grouped], grouper, None, epochs=10, verbose_every=1, 
                                       **other_kwargs)
        
            np.save(f"{myargs.save_path}/camelyon_{algorithm}_seed{seed}_{lamCIP}_unfreeze_{myargs.unfreeze_bn_and_augment}.npy", result)
        
        
        t = time.time()
        print(f"{algorithm} time: {t-t0:.2f} s\n")



