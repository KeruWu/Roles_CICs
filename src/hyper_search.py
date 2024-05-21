
## hyperparameter search

import numpy as np
import torch
import copy
import itertools
import math
import time
from collections import defaultdict
from src.method.alg_init import alg_init, alg_fit

## space of grid search

hyperparam_dict = {
    "DIP_mean":{
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    "DIP_MMD":{
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    "DIP_Pool_mean":{
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    "DIP_Pool_MMD":{
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_mean': {
        'lamCIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_MMD': {
        'lamCIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_LabelCorr_CIP_mean': {
        'lamCIP_A': [1e-2, .1, 1., 10., 100.],
        'lamCIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_LabelCorr_CIP_MMD': {
        'lamCIP_A': [1e-2, .1, 1., 10., 100.],
        'lamCIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_LabelCorr_CIP_MMD1': {
        'lamCIP_A': [1e-2, .1, 1., 10.],
        'lamCIP': [1e-2, .1, 1., 10.],
    },
    'CIP_LabelCorr_DIP_mean': {
        'lamCIP_A': [1e-2, .1, 1., 10., 100.],
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_LabelCorr_DIP_MMD': {
        'lamCIP_A': [1e-2, .1, 1., 10., 100.],
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_LabelCorr_DIP_MMD1': {
        'lamCIP_A': [1e-2, .1, 1., 10.],
        'lamDIP': [1e-2, .1, 1., 10.],
    },
    'CIP_JointCIPDIP_MMD':{
        'lamCIP_A': [1e-2, .1, 1., 10., 100.],
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_JointCIPDIP_MMD1':{
        'lamCIP_A': [1e-2, .1, 1., 10., 100.],
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_JointCIPDIP_MMD_pretrained':{
        'lamDIP': [1e-3, 1e-2, .1, 1., 10.],
    },
    'CIP_LabelCorr_JointCIPDIP_MMD':{
        'lamCIP_A': [1e-2, .1, 1., 10., 100.],
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    'CIP_LabelCorr_JointCIPDIP_MMD1':{
        'lamCIP_A': [1e-2, .1, 1., 10., 100.],
        'lamDIP': [1e-2, .1, 1., 10., 100.],
    },
    'IRM': {
        'lamIRM': [.1, 1., 10., 100., 1000., 10000.],
        'anneal_step': [0, 10, 100, 1000, 3000]
    },
    'VREx': {
        'lamVREx': [.1, 1., 10., 100., 1000., 10000.],
        'anneal_step': [0, 10, 100, 1000, 3000]
    },
    'groupDRO':{
        'group_weights_lr': [.01, .1, 1., 10.]
    },
}


def hyper_search(algorithm, device, model, task, dataloaders, grouper, optimizer,
                 nb_classes, n_epochs, verbose_every, M,
                 exp, seed, srcId, tarId, save_path, 
                 modelA, pretrained_modelA, #test_after_train, 
                 additional_dataloaders=None, additional_split_names=None, prop=[.05], **kwargs):
    t0 = time.time()
    if algorithm not in hyperparam_dict:
        print(f'Pass {algorithm}, no hyperparameter to search.\n')
        return 
    
    init_dict = copy.deepcopy(model.state_dict())
    if not pretrained_modelA:
        params = hyperparam_dict[algorithm]
    else:
        params = hyperparam_dict[algorithm+"_pretrained"]
    
    result_all = {}
    
    print(f'Hyperparameter search for {algorithm}')
    if algorithm not in ['CIP_JointCIPDIP_MMD1', 'CIP_LabelCorr_JointCIPDIP_MMD1', 'CIP_LabelCorr_CIP_MMD1', 'CIP_LabelCorr_DIP_MMD1']: 
        for s in itertools.product(*params.values()):
            alg_args = {}
            for i, key in enumerate(params.keys()):
                alg_args[key] = s[i]

            torch.manual_seed(seed)
            np.random.seed(seed)
            model.load_state_dict(init_dict)

            alg = alg_init(algorithm, device, model, optimizer, task, nb_classes, M, alg_args, 
                           modelA, pretrained_modelA, **kwargs)
            result = alg_fit(alg, algorithm, dataloaders, grouper, srcId[algorithm], tarId, n_epochs[algorithm], verbose_every,
                             additional_dataloaders=additional_dataloaders, additional_split_names=additional_split_names)

            for p in prop:
                _, alg_acc, _ = alg.predict_dataloader(dataloaders[tarId], prop=p)
                if len(prop) == 1:
                    result_all[s] = alg_acc
                else:
                    result_all[s] = result_all.get(s, []) + [alg_acc]

            cur_hyper = ""
            for i, key in enumerate(params.keys()):
                cur_hyper += key + " = " + str(s[i]) + ", "
            print(f'{cur_hyper:30} acc = {result_all[s]}')

    elif algorithm == 'CIP_LabelCorr_CIP_MMD1':
        for lamCIP_A in params['lamCIP_A']:
            for i, lamCIP in enumerate(params['lamCIP']):
                torch.manual_seed(seed)
                np.random.seed(seed)
                model.load_state_dict(init_dict)
                alg_args = {'lamCIP_A':lamCIP_A, 'lamCIP':lamCIP}
                
                if i == 0:
                    alg = alg_init(algorithm[:-1], device, model, optimizer, task, nb_classes, M, alg_args, 
                           modelA, pretrained_modelA, **kwargs)
                    result = alg_fit(alg, algorithm[:-1], dataloaders, grouper, srcId[algorithm[:-1]], tarId, n_epochs[algorithm[:-1]], 
                                     verbose_every, additional_dataloaders=additional_dataloaders, 
                                     additional_split_names=additional_split_names)

                    _, alg_acc, _ = alg.predict_dataloader(dataloaders[tarId], prop=prop)
                    
                    cip_model = copy.deepcopy(alg.modelA)
                    
                else:
                    
                    alg = alg_init(algorithm[:-1], device, model, optimizer, task, nb_classes, M, alg_args, 
                           modelA=cip_model, pretrained_modelA=True, **kwargs)
                    result = alg_fit(alg, algorithm[:-1], dataloaders, grouper, srcId[algorithm[:-1]], tarId, n_epochs[algorithm[:-1]], 
                                     verbose_every, additional_dataloaders=additional_dataloaders, 
                                     additional_split_names=additional_split_names)

                    _, alg_acc, _ = alg.predict_dataloader(dataloaders[tarId], prop=prop)
                    
                s = f'({lamCIP_A}, {lamCIP})'
                result_all[s] = alg_acc
            
                cur_hyper = f"lamCIP_A = {lamCIP_A}, lamCIP = {lamCIP},"
                print(f'{cur_hyper:30} acc = {result_all[s]:.3f}')
            
    
    else:
        for lamCIP_A in params['lamCIP_A']:
            for i, lamDIP in enumerate(params['lamDIP']):
                torch.manual_seed(seed)
                np.random.seed(seed)
                model.load_state_dict(init_dict)
                alg_args = {'lamCIP_A':lamCIP_A, 'lamDIP':lamDIP}
                
                if i == 0:
                    alg = alg_init(algorithm[:-1], device, model, optimizer, task, nb_classes, M, alg_args, 
                           modelA, pretrained_modelA, **kwargs)
                    result = alg_fit(alg, algorithm[:-1], dataloaders, grouper, srcId[algorithm[:-1]], tarId, n_epochs[algorithm[:-1]], 
                                     verbose_every, additional_dataloaders=additional_dataloaders, 
                                     additional_split_names=additional_split_names)

                    _, alg_acc, _ = alg.predict_dataloader(dataloaders[tarId], prop=prop)
                    
                    cip_model = copy.deepcopy(alg.modelA)
                    
                else:
                    
                    alg = alg_init(algorithm[:-1], device, model, optimizer, task, nb_classes, M, alg_args, 
                           modelA=cip_model, pretrained_modelA=True, **kwargs)
                    result = alg_fit(alg, algorithm[:-1], dataloaders, grouper, srcId[algorithm[:-1]], tarId, n_epochs[algorithm[:-1]], 
                                     verbose_every, additional_dataloaders=additional_dataloaders, 
                                     additional_split_names=additional_split_names)

                    _, alg_acc, _ = alg.predict_dataloader(dataloaders[tarId], prop=prop)
                    
                s = f'({lamCIP_A}, {lamDIP})'
                result_all[s] = alg_acc
            
                cur_hyper = f"lamCIP_A = {lamCIP_A}, lamDIP = {lamDIP},"
                print(f'{cur_hyper:30} acc = {result_all[s]:.3f}')
            
    t = time.time()
    print(f'time: {t-t0:.2f}s')
    np.save(f"{save_path}/hyper_{exp}_alg_{algorithm}_seed_{seed}.npy", result_all)


#     for s in result_all.keys():
#         print(f'{str(s):30}, acc = {result_all[s]:.3f}')

    
