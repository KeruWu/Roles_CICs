## Run experiments on SCM, MNIST, CelebA, and DomainNet


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
from src.method.alg_init import alg_init, alg_fit
from src import mmodel
from src import mdata
from src.config import dataset_defaults
from src.hyper_search import hyper_search

torch.backends.cudnn.deterministic = True

def compute_nll_loss(model, dataloader, weight, device):
    model.to(device)
    model.eval()
    loss = 0.
    n_samples = 0.
    criterion = torch.nn.CrossEntropyLoss(weight=weight, reduction='sum')
    with torch.no_grad():
        for data in dataloader:
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            loss += criterion(outputs, y).item()
            n_samples += len(y)
    return loss / n_samples

def calculate_Pi(model, dataloaders, device, nb_classes=2, binary_special=True):
    model.to(device)
    model.eval()
    N, M, L = 0, len(dataloaders), nb_classes
    with torch.no_grad():
        all_feats = {(m, y): [] for m in range(M) for y in range(L)}
        mu = {(m, y): [] for m in range(M) for y in range(L)}
        
        for m, dataloader in enumerate(dataloaders):
            feats = []
            labels = []
            for data in dataloader:
                x, y = data[0].to(device), data[1].to(device)
                feat = model.featurizer(x)
                if binary_special:
                    feats.append(feat[:,[0]] - feat[:,[1]])
                else:
                    feats.append(feat)
                labels.append(y)
                N += len(y)
            feats = torch.cat(feats, dim=0)
            labels = torch.cat(labels)
            for y in range(L):
                idx = labels == y
                all_feats[(m, y)] = feats[idx]
                mu[(m,y)] = feats[idx].mean(dim=0)
        Delta = {}
        Sigma, Mhat = 0, {y:0 for y in range(L)}
        Pihat = np.zeros(L)
        for y in range(L):
            for m in range(1, M):
                Delta[(m,y)] = mu[(m, y)] - mu[(0, y)]
                Mhat[y] += (Delta[(m, y)]**2).sum()
                centered_feat = all_feats[(m, y)] - mu[(m, y)].reshape(1, -1)
                Sigma += torch.matmul(centered_feat.T, centered_feat)
            Mhat[y] /= M-1
        Sigma /= N-M*L
        Sig_inv = torch.inverse(Sigma)
        
        for y in range(L):
            for m in range(1, M):
                Pihat[y] += torch.matmul(torch.matmul(Delta[(m,y)].reshape(1,-1), Sig_inv), Delta[(m,y)]).detach().cpu().numpy()
            Pihat[y] /= M-1
    return Pihat
            

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs='+', type=int, default=[0], help="seeds in experiments")
    parser.add_argument("--exp", type=str, default=None, help="Which experiment to do.")
    parser.add_argument("--hyper_search", type=int, default=0, help="Search hyperparameters or not")
    parser.add_argument("--hyper_prop", nargs='+', type=float, default=[.05])
    parser.add_argument('--algs', nargs='+', type=str, default=None, help="list of algorithms to test, if not provided, use default settings in the config file")
    parser.add_argument('--data_root', type=str, default='../data', help="path to data")
    parser.add_argument('--save_path', type=str, default='../results', help="path to save results")
    parser.add_argument('--save_model', type=int, default=0, help='save model state dict or not')
    parser.add_argument('--save_result', type=int, default=1, help='save result or not')
    parser.add_argument("--save_each_alg", type=int, default=0, help='save result for each algorithm')
    parser.add_argument("--save_loss", type=int, default=0, help='save the cross entropy loss')
    parser.add_argument("--calculate_Pi", type=int, default=0, help='calculate Pi_phi(y)')
    
    # hyperparameter arguments
    parser.add_argument("--lamCIP", type=float, help="lamCIP hyperparameter")
    parser.add_argument("--lamCIP_A", type=float, help="lamCIP_A hyperparameter")
    parser.add_argument("--lamDIP", type=float, help="lamDIP hyperparameter")
    parser.add_argument("--lamIRM", type=float, help="lamIRM hyperparameter")
    parser.add_argument("--lamVREx", type=float, help="lamVREx hyperparameter")
    parser.add_argument("--group_weights_lr", type=float, help="group_weights_lr hyperparameter")
    parser.add_argument("--anneal_step", type=float, help="anneal_step hyperparameter")    
    
    myargs = parser.parse_args()
    
    ## default dataset config
    config = dataset_defaults[myargs.exp]
    data_root = myargs.data_root
    
    ## common configs shared by algorithms
    algs = config['algs'] if myargs.algs == None else myargs.algs
    optimizer = config['optimizer']
    nb_classes = config['nb_classes']
    n_epochs = defaultdict(lambda: config['n_epochs'])
    srcId = defaultdict(lambda: [0])
    verbose_every = config['verbose_every'] if 'verbose_every' in config else 1
    task = config['task'] if 'task' in config else 'classification'
    
    optimizer_args = {}
    optimizer_args['lr'] = config['lr']
    if 'weight_decay' in config:
        optimizer_args['weight_decay'] = config['weight_decay']
    if 'momentum' in config:
        optimizer_args['momentum'] = config['momentum']
    
    for algorithm in algs:
        ## specify any special configs (different from above) for certain algorithms here
        if algorithm in config:
            if 'n_epochs' in config[algorithm]:
                n_epochs[algorithm] = config[algorithm]['n_epochs']
            if 'srcId' in config[algorithm]:
                srcId[algorithm] = config[algorithm]['srcId']
    M = config['M']
    tarId = M-1 if 'tarId' not in config else config['tarId']
    
    ## custom configs for different experiments
    if myargs.exp[:3] == 'SCM':
        n = config['n']
        d = config['d']
    
    ## default settings for some parameters
    grouper=None
    modelA = None
    pretrained_modelA = False
    additional_dataloaders=None
    additional_split_names=None
    
    
    
    for seed in myargs.seeds:
        
        print(f'seed = {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if myargs.exp[:3] == 'SCM':
            if 'binary' not in myargs.exp:
                dataloaders = getattr(mdata.simu, myargs.exp[:5])(M=M, n=n, d=d)
                model = mmodel.Linear(d, nb_classes).to(device)
            elif 'linear' not in myargs.exp:
                dataloaders = getattr(mdata.simu, myargs.exp[:10])(M=M, n=n, d=d)
                model = mmodel.MLP(d, [d], nb_classes).to(device)
            else:
                dataloaders = getattr(mdata.simu, myargs.exp[:10])(M=M, n=n, d=d)
                model = mmodel.Linear(d, nb_classes).to(device)
        
        
        elif myargs.exp[:5] == 'MNIST':
            dataloaders = mdata.mnist.mnist_perturb_preset(perturb=myargs.exp[6:], M=M, datapath=data_root)
            model = mmodel.LeNet5(nb_classes).to(device)
        
        elif myargs.exp[:6] == 'CelebA':
            image_path = data_root + '/CelebA/img_align_celeba'
            attr_path = data_root + '/CelebA/attr/list_attr_celeba.txt'    
            dataloaders = mdata.celeba.celeba_perturb(perturb = myargs.exp,
                                  image_path = image_path,
                                  attr_path = attr_path)
            model = mmodel.CNN_CelebA(nb_classes = 2).to(device)
            
        elif myargs.exp == 'DomainNet':
            dataloaders, _ = mdata.domainnet.domainNet_loaders(
                directory=data_root + '/domainNet',
                divs=['Furniture', 'Mammal'],
                batch_size=16
                )    
            model = mmodel.ResNet(nb_classes=2).to(device)        
            
        else:
            raise ValueError('Invalid experiment.')
            
        init_dict = copy.deepcopy(model.state_dict())
        
        
        
        if myargs.hyper_search:
            
            for algorithm in algs:
                model.load_state_dict(init_dict)
                hyper_search(algorithm, device, model, task, dataloaders, grouper, optimizer,
                             nb_classes, n_epochs, verbose_every, M, 
                             myargs.exp, seed, srcId, tarId, myargs.save_path, 
                             modelA, pretrained_modelA, #test_after_train, 
                             additional_dataloaders=additional_dataloaders, additional_split_names=additional_split_names,
                             prop=myargs.hyper_prop,
                             **optimizer_args)
                
        else:
            
            results_all = np.zeros((len(algs), M+1))
            loss_all = np.zeros((len(algs), M))
            Pihat = np.zeros((len(algs), nb_classes))
            t_all = np.zeros(len(algs))
            t0 = time.time()
            
            if not myargs.save_each_alg:
                save_file = f"{myargs.save_path}/{myargs.exp}_seed{seed}_.npy"
            else:
                save_file = f"{myargs.save_path}/{myargs.exp}_seed{seed}_{algs}_.npy"            
            
            for k, algorithm in enumerate(algs):
                print(algorithm)
                torch.manual_seed(seed)
                np.random.seed(seed)
                model.load_state_dict(init_dict)
                
                if 'each_seed_different' in config and algorithm in config:
                    alg_args = config[algorithm]
                    for key in alg_args:
                        if key != 'srcId':
                            alg_args[key] = alg_args[key][seed-1]
                    print(alg_args)
                else:
                    alg_args = config[algorithm] if algorithm in config else None
                    
                if alg_args is not None:
                    for key in alg_args.keys():
                        myargs_value = getattr(myargs, key, None)
                        if myargs_value is not None:
                            alg_args[key] = myargs_value                    
                
                alg = alg_init(algorithm, device, model, optimizer, task, nb_classes, M, alg_args, 
                               modelA, pretrained_modelA, **optimizer_args)
                result = alg_fit(alg, algorithm, dataloaders, grouper, srcId[algorithm], tarId, n_epochs[algorithm], verbose_every,
                                 additional_dataloaders=additional_dataloaders, additional_split_names=additional_split_names)
                
                
                #if myargs.save_model:
                if myargs.exp == 'SCM_3' or 'SCM_binary' in myargs.exp:
                    torch.save(alg.model.state_dict(), f"{myargs.save_path}/{myargs.exp}_{algorithm}_model_seed{seed}_.pt")
                
                
                
                train_correct = 0
                train_sample = 0


                for i in range(M):
                    ypreds, acc, correct = alg.predict_dataloader(dataloaders[i])
                    results_all[k, i] = acc
                    print("env %d, method %-20s" %(i, algorithm), "accuracy %.3f" % acc)
                    if i!= tarId:
                        train_correct += correct
                        train_sample += len(ypreds)
                results_all[k, -1] = train_correct / float(train_sample)

                if myargs.save_loss:
                    weight = alg.w_corrected if hasattr(alg, 'w_corrected') else None
                    #print(weight)
                    for i in range(M):
                        loss_all[k, i] = compute_nll_loss(alg.model, dataloaders[i], 
                                                          weight=weight if i != tarId else None,
                                                          device=device)
                    print(loss_all[k,:])

                if myargs.calculate_Pi:
                    Pihat[k] = calculate_Pi(model, dataloaders[:-1], device, nb_classes=2) 
                    #[dataloaders[i] for i in range(len(dataloaders)) if i != tarId]
                    
                t = time.time()
                t_all[k] = t - t0
                print(f"{algorithm} time: {t-t0:.2f} s\n")
                t0 = t
                

            print('\n===========================================\n', 'summary (each domain):')
            for i in range(M):
                for k, algorithm in enumerate(algs):
                    print("env %d, method %-20s" %(i, algorithm), "accuracy %.3f" % results_all[k, i])
                print()

            print('\n===========================================\n', 'summary (source & target):')
            for k, algorithm in enumerate(algs):
                print(f"{algorithm+':':20} src acc: {results_all[k,-1]:.3f}, tar acc: {results_all[k,-2]:.3f}, time: {t_all[k]:.2f} s")

            if myargs.save_result:
                np.save(save_file, results_all)
            if myargs.save_loss:
                print(loss_all)
                np.save(f"{myargs.save_path}/{myargs.exp}_seed{seed}_loss.npy", loss_all)
            if myargs.calculate_Pi:
                np.save(f"{myargs.save_path}/{myargs.exp}_seed{seed}_Pi.npy", Pihat)
        
            
        