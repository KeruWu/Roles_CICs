import numpy as np
import torch

class CMShiftCorrector():
    """
    Class to correct label shift using confusion matrix.
    """
    def __init__(self, device, nb_classes, correct_each_domain=False):
        self.device = device
        self.nb_classes = nb_classes
        self.correct_each_domain = correct_each_domain


    def __call__(self, s_loader, t_loader, model):
        """
        Estimate the unknown target label distribution by inverting linear system.
        Parameters
        ----------
        s_loader : torch.utils.data.DataLoader
            Dataloader for source domain (a list of dataloaders)

        t_loader : torch.utils.data.DataLoader
            Dataloader for target domain

        model : Pytorch model
            Model calibrated used on label shift correction algorithm
        """

        q = self._get_q_vector(t_loader, model)
        nb_domains = len(s_loader)
        
        if self.correct_each_domain:
            C = self._get_confusion_matrix(s_loader, model, get_for_each_domain=True)
            w = np.zeros(nb_domains, self.nb_classes)
            for i in range(nb_domains):
                w[i] = np.linalg.solve(C, q)
            return torch.Tensor(w).to(self.device)
        
        else:
            C = self._get_confusion_matrix(s_loader, model, get_for_each_domain=False)
            try:
                w = np.matmul(np.linalg.inv(C),  q)
                w[np.where(w < 0)[0]] = 0
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    print('Cannot compute using matrix inverse due to singlar matrix, using psudo inverse')
                    w = np.matmul(np.linalg.pinv(C), q)
                    w[np.where(w < 0)[0]] = 0
                else:
                    raise RuntimeError("Unknown error")
            #w = np.linalg.solve(C, q)
            w = torch.Tensor(w).to(self.device)
            return w
            #return w.repeat(nb_domains, 1)


    def _get_q_vector(self, dataloader, model):
        q = np.zeros(self.nb_classes)
        for _, data in enumerate(dataloader):
            inputs = data[0].to(self.device)
            _, y_preds = torch.max(model(inputs).data, 1)
            for i in range(self.nb_classes):
                q[i] += torch.sum(y_preds == i).item()
        q /= q.sum()
        return q


    def _get_confusion_matrix(self, dataloaders, model, get_for_each_domain=True):
        
        if get_for_each_domain:
            C = np.zeros((len(dataloaders), self.nb_classes, self.nb_classes))
            for i, dataloader in enumerate(dataloaders):
                for data in dataloader:
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)
                    _, ypreds = torch.max(model(inputs).data, 1)
                    for idx in range(len(inputs)):
                        row = y_preds[idx].item()
                        col = labels[idx].item()
                        C[i, row, col] += 1                    
                C[i] /= C[i].sum()
            return C
        
        else:
            C = np.zeros((self.nb_classes, self.nb_classes))
            for dataloader in dataloaders:
                for _, data in enumerate(dataloader):
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)
                    _, y_preds = torch.max(model(inputs).data, 1)
                    for idx in range(len(inputs)):
                        row = y_preds[idx].item()
                        col = labels[idx].item()
                        C[row,col] += 1
            C /= C.sum()
            return C