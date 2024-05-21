## JointDIP on MNIST 3 with different lambdas

import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys

sys.path.append("..")
import src.method
import src.mdata
import copy

from src import mmodel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seeds = range(1, 11)
M=6
srcInd = 0
nb_classes=2
source = [i for i in range(M-1)]
target = M-1

lambdas = [1e-2, 1e-1, 1., 10., 100.]
for seed in seeds:
    dataloaders = src.mdata.mnist.mnist_perturb_preset(perturb='rotation_3', M=6, datapath='../data')
    print(f'Seed = {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    mCIP = src.method.CIP(device=device, model=mmodel.LeNet5(nb_classes).to(device),
                             lamCIP=.1, discrepType='MMD', nb_classes=2, 
                             loss_type='CrossEntropyLoss', optimizer='Adam', lr=1e-3)
    mCIP.fit(dataloaders[:-1], None, None, epochs=20, verbose_every=20)

    ypred_tar_CIP, acc_tar_CIP, _ = mCIP.predict_dataloader(dataloaders[target])
    ypred_src_CIP, acc_src_CIP, _ = mCIP.predict_dataloader(dataloaders[srcInd])
    
    print(f'CIP tar acc {acc_tar_CIP:.3f}')
    print(f'CIP src acc {acc_src_CIP:.3f}')
    

    for lamJointDIP in lambdas:

        model=mmodel.LeNet5(nb_classes).to(device)
        mjointDIP = src.method.CIP_JointCIPDIP(device=device, model=model, lamDIP=lamJointDIP, 
                                                  modelA=None, lamCIP_A=1., discrepType='MMD', pretrained_modelA=False,
                                                  nb_classes=2, loss_type='CrossEntropyLoss', optimizer='Adam', lr=1e-3)
        mjointDIP.fit(dataloaders, None, [0], M-1, epochs=20, verbose_every=20)

        ypred_tar_jointDIP, acc_tar_jointDIP, _ = mjointDIP.predict_dataloader(dataloaders[target])
        ypred_src_jointDIP, acc_src_jointDIP, _ = mjointDIP.predict_dataloader(dataloaders[srcInd])
        
        print(f'jointDIP tar acc: {acc_tar_jointDIP:.3f}')
        print(f'jointDIP src acc: {acc_src_jointDIP:.3f}')


        disagreement_tar = torch.sum(ypred_tar_CIP!=ypred_tar_jointDIP)/ypred_tar_jointDIP.shape[0]
        disagreement_src = torch.sum(ypred_src_CIP!=ypred_src_jointDIP)/ypred_src_jointDIP.shape[0]

        print(f"disagreement tar: {disagreement_tar:.3f}")
        print(f"disagreement src, {disagreement_src:.3f}")

        jointDIP_tar_risk_lower_bound = (1-acc_src_jointDIP) - 2 * (1-acc_src_CIP) + disagreement_tar - disagreement_src

        print(f"jointDIP target accuracy upper bound, {1-jointDIP_tar_risk_lower_bound:.3f}",)
        print(f"Should >= {acc_tar_jointDIP:.3f}\n") 
        
        result = torch.tensor([1-jointDIP_tar_risk_lower_bound, acc_tar_jointDIP, disagreement_tar, disagreement_src,
                            acc_src_jointDIP, acc_tar_CIP, acc_src_CIP]).detach().cpu().numpy()
        
        save_file = f'../results/DIPfail_MNIST3_seed_{seed}_JointDIP_{lamJointDIP}_1.0_.npy'
        np.save(save_file, result)


