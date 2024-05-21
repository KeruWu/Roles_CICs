import torch
import numpy as np
import sys
from functools import partial
from tqdm import tqdm
# to avoid printing new lines
#tqdm = partial(tqdm, position=0, leave=True)
from time import sleep
from collections import defaultdict

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

class Trainer():
    """
    Class to handle training of model.
    """
    def __init__(self, model, loss, optimizer, device=None, alg=None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.alg = alg

    def train(self, dataloaders, grouper=None, tarId=None, epochs=10, verbose_every=1, 
              additional_dataloaders=None, additional_split_names=None, save_pred_path=None, seed=None, 
              val_split_name='val', **kwargs):
        """Train the model

        Args:
            dataloaders: list of dataloaders
            grouper: if None, each dataloader corresponds to each domain,
                     otherwise, grouper.metadata_to_group gives the group id (specially designed for WILDS).
            tarId (int): which index of dataloaders corresponds to the target domain. 
                         If None: target domain is the last dataloader.
            epochs (int): number of epochs for training.
            verbose_every (int): use tqdm / check accuracy every verbose_every epoch.
            kwargs (optional): {additional_dataloaders: dataloaders used to check accuracy during training 
                                                        (every verbose_every epoch)
                                additional_split_names: corresponding name of additional_dataloaders}
        """
        self.model.train()
        self.grouper = grouper
        
        result = defaultdict(list)
        result['verbose_every'] = verbose_every
        
        
        
        if tarId is not None and tarId not in [len(dataloaders)-1, -1]:
            print('changing dataloader order to make the last one as target domain')
            dataloaders[tarId], dataloaders[-1] = dataloaders[-1], dataloaders[tarId]
        
        
        if additional_dataloaders is not None:
            assert len(additional_dataloaders)==len(additional_split_names), f"number of additional_dataloaders should equal to additional_split_names, got: {len(additional_dataloaders)} and {len(additional_split_names)}"
        
        if save_pred_path is not None:
            best_model = None
            best_pred = None
            best_val_acc = 0.
            best_y_pred_all = {}
            
        for epoch in range(epochs):
            verbose = (epoch+1)%verbose_every == 0
            if self.alg is not None:
                epoch_loss = self._train_epoch_generic(dataloaders, epoch, verbose, **kwargs)
            else:
                epoch_loss = self._train_epoch_losspenalty(dataloaders, epoch, verbose, **kwargs)
            result['train_losses'].append(epoch_loss)
            if verbose and additional_dataloaders is not None:
                y_pred_all = {}
                for dataloader, split_name in zip(additional_dataloaders, additional_split_names):
                    
                    if self.grouper is None:
                        ypreds_final, accuracy, correct = self.predict(dataloader)
                        result[split_name].append(accuracy)
                        print(f'{split_name}: {accuracy:.3f}')
                    else:
                        res, all_y_pred = self.predict_wilds(dataloader)
                        result[split_name].append(res)
                        y_pred_all[split_name] = all_y_pred
                        print(f'{split_name}: {res[1]}')
                    
                if save_pred_path is not None:
                    val_acc = result[val_split_name][-1] if self.grouper is None else result[val_split_name][-1][0]['acc_avg']
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_y_pred_all = y_pred_all
                        for split_name in best_y_pred_all:
                            np.savetxt(f"{save_pred_path}/camelyon17_split:{split_name}_seed:{seed}_epoch:best_pred.csv", best_y_pred_all[split_name].astype(int), fmt='%i', delimiter='\t')

        return result

    def _train_epoch_losspenalty(self, dataloaders, epoch, verbose, **kwargs):
        """Train the model for one epoch for methods that update model after loss.backward()

        Args:
            dataloaders: list of dataloaders
            epoch (int): current epoch number
            verbose (bool): use tqdm or not
            kwargs (optional): optimizer arguments
        """
        self.model.train()
        epoch_loss = 0.
        iterator = self._create_iterator(dataloaders, verbose)
        batch_idx = 0

        for data in iterator:
            if self.grouper is not None:
                metadata = data[0][2]
                groups = self.grouper.metadata_to_group(metadata)
            else:
                groups = None

            loss = self.loss(data, self.model, groups, **kwargs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            batch_idx += 1
            if verbose:
                iterator.set_description(f"Epoch {epoch}") 
                iterator.set_postfix(epoch_loss=f"{epoch_loss / batch_idx:.8f}") #batch_loss=f"{loss.item():.4f}") 


        mean_epoch_loss = epoch_loss / batch_idx
        return mean_epoch_loss
    
    def _train_epoch_generic(self, dataloaders, epoch, verbose, **kwargs):
        """Train the model for one epoch for methods that update model with a customized function alg.process_batch
           (e.g. can be used for meta-learning)
        Args:
            dataloaders: list of dataloaders
            epoch (int): current epoch number
            verbose (bool): use tqdm or not
            kwargs (optional): optimizer arguments
        """
        self.model.train()
        epoch_loss = 0.
        iterator = self._create_iterator(dataloaders, verbose)
        batch_idx = 0

        for data in iterator:
            
            if self.grouper is not None:
                metadata = data[0][2]
                groups = self.grouper.metadata_to_group(metadata)
            else:
                groups = None
            
            loss = self.alg.process_batch(data, groups, **kwargs)
            epoch_loss += loss.item()
            
            batch_idx += 1
            if verbose:
                iterator.set_description(f"Epoch {epoch}")
                iterator.set_postfix(batch_loss=f"{loss.item():.4f}") #epoch_loss=f"{epoch_loss / batch_idx:.4f}") 

        mean_epoch_loss = epoch_loss / batch_idx
        return mean_epoch_loss

    
    def _create_iterator(self, dataloaders, verbose):

        n_envs = len(dataloaders)
        num_samples = np.zeros(n_envs)
        for i in range(n_envs):
            size_i = len(dataloaders[i])
            num_samples[i] = size_i
        max_sampleInd = np.argmax(num_samples)
        dataloaders_cycled = [cycle(dataloaders[i]) if i!=max_sampleInd else dataloaders[i] for i in range(n_envs)]
        iterator = tqdm(zip(*dataloaders_cycled), total=int(max(num_samples))) if verbose else zip(*dataloaders_cycled)
        return iterator

    def predict(self, dataloader, prop=1.):
        self.model.eval()
        correct = 0
        if prop <= 1.:
            N = int(len(dataloader.dataset) * prop)
        else:
            N = int(prop)
        ypreds_final = torch.zeros((N, 1))
        pos = 0
        with torch.no_grad():
            #for data in tqdm(dataloader):
            for data in dataloader:
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                outputs = self.model(x)

                ypreds = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                n = min(len(ypreds), N-pos)
                ypreds = ypreds[:n]
                ypreds_final[pos:pos+n] = ypreds
                pos += len(ypreds)
                correct += ypreds.eq(y[:n].data.view_as(ypreds)).cpu().sum()
                if pos >= N:
                    break
        accuracy = correct / float(N)
        return ypreds_final, accuracy, correct
    
    
    def predict_wilds(self, dataloader):
        self.model.eval()
        all_y_pred = []
        all_y_true = []
        all_meta = []
        with torch.no_grad():
            for x, y_true, metadata in tqdm(dataloader):
                x = x.to(self.device)
                y_pred = self.model(x).data.max(1)[1].detach().cpu()
                all_y_pred.append(y_pred)
                all_y_true.append(y_true)
                all_meta.append(metadata)
            all_y_pred = torch.cat(all_y_pred)
            all_y_true = torch.cat(all_y_true)
            all_meta = torch.cat(all_meta)
            res = dataloader.dataset.eval(all_y_pred, all_y_true, all_meta)
        return res, all_y_pred.numpy()
    