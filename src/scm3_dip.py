## DIP on SCM 3 with different lambdas and alphas

import numpy as np
import torch
import torchvision

import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys

sys.path.append("..")
import src.method
import src.mdata
import copy

from src import mmodel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.)
myargs = parser.parse_args()

proportion = 1-myargs.alpha


    
def predict(dataloader, model):
    model.eval()
    correct = 0
    ypreds_final = torch.zeros((len(dataloader.dataset), 1))
    y_all = []
    pos = 0
    outputs_all = []
    #for batch_idx, data in enumerate(tqdm(dataloader)):
    for batch_idx, data in enumerate(dataloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)
        outputs = model(inputs)
        ypreds = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        ypreds_final[pos:pos+len(ypreds)] = ypreds
        yprob = np.exp(outputs.detach().cpu().numpy())
        yprob /= yprob.sum(axis=1).reshape(-1, 1)
        outputs_all.append(yprob)
        y_all.append(labels.detach().cpu().numpy().reshape(-1))
        pos += len(ypreds)
        correct += ypreds.eq(labels.data.view_as(ypreds)).cpu().sum()

    accuracy = correct / np.float(len(ypreds_final))
    return ypreds_final, accuracy, correct, np.concatenate(outputs_all), np.concatenate(y_all)


seeds = range(1, 11)
M=12
n=1000
d=18
nb_classes=2
srcInd = 0
source = [i for i in range(M-1)]
target = M-1
lambdas = [.01, .1, 1., 10., 100.]
discrepType = 'mean'

for seed in seeds:
    print(f'Seed = {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataloaders = src.mdata.simu.SCM_3(M=M, n=n, d=d)
    mCIP = src.method.CIP(device=device, model=mmodel.Linear(d, nb_classes).to(device),
                             lamCIP=10., discrepType='mean', nb_classes=2, 
                             loss_type='CrossEntropyLoss', optimizer='Adam', lr=1e-2)
    mCIP.fit(dataloaders[:-1], None, None, epochs=50, verbose_every=51)

    ypred_tar_CIP, acc_tar_CIP, _, yprob_tar_CIP, y_tar_all = predict(dataloaders[target], mCIP.model)
    ypred_src_CIP, acc_src_CIP, _, yprob_src_CIP, y_src_all = predict(dataloaders[srcInd], mCIP.model)
    
    prob_tar_CIP = np.max(yprob_tar_CIP, axis=1)
    prob_src_CIP = np.max(yprob_src_CIP, axis=1)
    
    thres_CIP = np.quantile(prob_tar_CIP, 1-proportion)
    indx_tar = prob_tar_CIP > thres_CIP
    indx_src = prob_src_CIP > thres_CIP
    prop = indx_tar.sum() / len(ypred_tar_CIP)
    
    
    print(f'CIP tar acc {acc_tar_CIP:.3f}')
    print(f'CIP src acc {acc_src_CIP:.3f}')
    
    print(f'{proportion*100:.2f}% data in region: pred prob > {thres_CIP}')
    
    acc_tar_CIP_A = (ypred_tar_CIP.numpy().reshape(-1)[indx_tar]==y_tar_all[indx_tar]).sum()/len(y_tar_all[indx_tar])
    acc_src_CIP_A = (ypred_src_CIP.numpy().reshape(-1)[indx_src]==y_src_all[indx_src]).sum()/len(y_src_all[indx_src])
    
    print(f'CIP tar acc on A: {acc_tar_CIP_A:.3f}')
    print(f'CIP src acc on A: {acc_src_CIP_A:.3f}')
    
    print()
    
    mDIP = []
    for lamDIP in lambdas:
        print(f'lamDIP = {lamDIP}')
        mDIP = src.method.DIP(device=device, model=mmodel.Linear(d, nb_classes).to(device),
                                 lamDIP=lamDIP, discrepType='mean', nb_classes=2, loss_type='CrossEntropyLoss',
                                 optimizer='Adam', lr=1e-2)
        mDIP.fit([dataloaders[0], dataloaders[-1]], None, -1, epochs=50, verbose_every=51)

        ypred_tar_DIP, acc_tar_DIP, _, yprob_tar_DIP, y_tar_all = predict(dataloaders[target], mDIP.model)
        ypred_src_DIP, acc_src_DIP, _, yprob_src_DIP, y_src_all = predict(dataloaders[srcInd], mDIP.model)
        
        print(f'DIP tar acc: {acc_tar_DIP:.3f}')
        print(f'DIP src acc: {acc_src_DIP:.3f}')


        disagreement_tar = torch.sum(ypred_tar_CIP!=ypred_tar_DIP)/ypred_tar_DIP.shape[0]
        disagreement_src = torch.sum(ypred_src_CIP!=ypred_src_DIP)/ypred_src_DIP.shape[0]

        print(f"disagreement tar: {disagreement_tar:.3f}")
        print(f"disagreement src, {disagreement_src:.3f}")

        DIP_tar_risk_lower_bound = (1-acc_src_DIP) - 2 * (1-acc_src_CIP) + disagreement_tar - disagreement_src

        print(f"DIP target accuracy upper bound, {1-DIP_tar_risk_lower_bound:.3f}",)
        print(f"Should >= {acc_tar_DIP:.3f}\n") 
        
        result = torch.tensor([1-DIP_tar_risk_lower_bound, acc_tar_DIP, disagreement_tar, disagreement_src,
                            acc_src_DIP, acc_tar_CIP, acc_src_CIP]).detach().cpu().numpy()
        
        
        print(f"In region CIP pred prob > {thres_CIP}")
        
        acc_tar_DIP_A = (ypred_tar_DIP.numpy().reshape(-1)[indx_tar]==y_tar_all[indx_tar]).sum()/len(y_tar_all[indx_tar])
        acc_src_DIP_A = (ypred_src_DIP.numpy().reshape(-1)[indx_src]==y_src_all[indx_src]).sum()/len(y_src_all[indx_src])
        
        print(f'DIP tar acc on A: {acc_tar_DIP_A:.3f}')
        print(f'DIP src acc on A: {acc_src_DIP_A:.3f}')
        
        disagreement_tar_A = torch.sum(ypred_tar_CIP[indx_tar]!=ypred_tar_DIP[indx_tar])/len(y_tar_all[indx_tar])
        disagreement_src_A = torch.sum(ypred_src_CIP[indx_src]!=ypred_src_DIP[indx_src])/len(y_src_all[indx_src])

        print(f"disagreement tar on A: {disagreement_tar_A:.3f}")
        print(f"disagreement src on A, {disagreement_src_A:.3f}")

        DIP_tar_risk_lower_bound_A = (1-acc_src_DIP_A) - 2 * (1-acc_src_CIP_A) + disagreement_tar_A - disagreement_src_A

        print(f"DIP target accuracy upper bound on A, {1-DIP_tar_risk_lower_bound_A:.3f}",)
        print(f"Should >= {acc_tar_DIP_A:.3f}  on A \n") 
        
        result_A = torch.tensor([1-DIP_tar_risk_lower_bound_A, acc_tar_DIP_A, disagreement_tar_A, disagreement_src_A,
                            acc_src_DIP_A, acc_tar_CIP_A, acc_src_CIP_A]).detach().cpu().numpy()
        
        
        save_file = f'../results/DIPfail_SCM3_seed_{seed}_DIP_{lamDIP}_{proportion}_.npy'
        np.save(save_file, result_A)
        