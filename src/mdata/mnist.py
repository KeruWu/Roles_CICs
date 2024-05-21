"""Generate MNIST perturbed environments
"""
import torch
import torchvision
import numpy as np

from typing import Any, Callable, Optional, Tuple
from PIL import Image
import os


class BarTransform(object):
    """Custom bar transformer"""

    def __init__(self, zero_trans='h', one_trans='v'):
        self.zero_trans = zero_trans
        self.one_trans = one_trans

        patch_hbar = torch.zeros(1, 28, 28)
        patch_vbar = torch.zeros(1, 28, 28)

        offset = 10
        initpos = 2
        sqsize = 16

        patch_hbar[0, (initpos+offset+sqsize//2):(initpos+offset+sqsize//2+6),#3),
                                (initpos-1):(initpos+sqsize-1)] = 3.25

        #     patch_hbar[0, 18:24, 4:10] = 3.25

        patch_vbar[0, initpos-1:(initpos+sqsize-1),
                      (initpos+sqsize//2+offset):(initpos+sqsize//2+offset+3)] = 3.25
        self.patch_hbar = patch_hbar
        self.patch_vbar = patch_vbar

    def __call__(self, image, label, do_lab_transform):
        if do_lab_transform:
        # Not every data point will get the label dependent transform
            if label == 0: #sum(label == i for i in [0,1,2,3,4]):
                if self.zero_trans == 'h':
                    image = torch.max(image, self.patch_hbar)
                elif self.zero_trans == 'v':
                    image = torch.max(image, self.patch_vbar)
            else: # label == 1
                if self.one_trans == 'h':
                    image = torch.max(image, self.patch_hbar)
                elif self.one_trans == 'v':
                    image = torch.max(image, self.patch_vbar)
        else:
        # do the reverse transform
            if label == 1: #sum(label == i for i in [5,6,7,8,9]):
                if self.zero_trans == 'h':
                    image = torch.max(image, self.patch_hbar)
                elif self.zero_trans == 'v':
                    image = torch.max(image, self.patch_vbar)
            
            else: # label == 0
                if self.one_trans == 'h':
                    image = torch.max(image, self.patch_hbar)
                elif self.one_trans == 'v':
                    image = torch.max(image, self.patch_vbar)
        # else do nothing

        return image, label



class MNISTlab(torchvision.datasets.MNIST):
    """Custom MNIST dataset 
    which allows for label dependent transform
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        lab_dep_transform: Optional[Callable] = None,
        do_lab_transform_prob: float = 0,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        self.lab_dep_transform = lab_dep_transform
        self.do_lab_transform_prob = do_lab_transform_prob
        if (lab_dep_transform is not None) and (do_lab_transform_prob > 0):
            self.do_lab_transforms = (np.random.rand(self.targets.shape[0]) < do_lab_transform_prob)
      

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if (self.lab_dep_transform is not None) and (self.do_lab_transform_prob > 0):
            # index is needed because label flipping prob is pre-generated
            img, target = self.lab_dep_transform(img, target, self.do_lab_transforms[index])

        return img, target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')

    
    
def _subset_dataset(dataset, interY, interYprop=0.5, subsetProp=0.9):
    """Take a dataset with data and targets
    return a subset with digits in digits,
    with a designed Y label distribution

    Args:
      dataset (torchvision.datasets.MNIST): dataset
      interY (bool): whether to have Y label shift
      interYprop (float, optional): Defaults to 0.5.
      subsetProp (float, optional): 
      randomly remove 1-that proportion of data. Defaults to 0.9.

    Returns:
      modified dataset
    """

    idx0 = sum(dataset.targets == i for i in [0,1,2,3,4]).bool()
    idx1 = sum(dataset.targets == i for i in [5,6,7,8,9]).bool()

    dataset.targets[idx0] = 0
    dataset.targets[idx1] = 1

    if interY:
    # keep interYprop proportion of digit 0
        idx = idx1 | (idx0 & (torch.rand_like(dataset.targets.float()) < interYprop))
    else:
        idx = idx1 | idx0
    
    idx = idx & (torch.rand_like(dataset.targets.float()) < subsetProp)
  
    # subset the dataset
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    return dataset


def _transformer_picker(perturbType, M):
    """
    pick a transfomer 
    """
    if perturbType == 'classic':
        transformers = {}
        for m in range(M):
            transformers[m] = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1306,), (0.3081,)),
                ])
    elif perturbType == 'whitepatch':
    # prepare the patches
        patches = []
        offset = 10
        initpos = 2
        sqsize = 16
        for m in range(M):
            apatch = torch.zeros(1, 28, 28)
            apatch[0, (initpos-2*m+offset):initpos+sqsize-2*m+offset, (initpos-2*m+offset):initpos+sqsize-2*m+offset] = 3.25

            patches.append(apatch)

        transformers = {}
        for m in range(M):
            transformers[m] = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1306,), (0.3081,)),
                 torchvision.transforms.Lambda((lambda y: lambda x: torch.max(x, patches[y]))(m)),
                ])
    elif perturbType == 'smallwhitepatch':
        # prepare the patches
        patches = []
        offset = 10
        initpos = 2
        sqsize = 12
        for m in range(M):
            apatch = torch.zeros(1, 28, 28)
            apatch[0, (initpos-2*m+offset):initpos+sqsize-2*m+offset, (initpos-2*m+offset):initpos+sqsize-2*m+offset] = 3.25

            patches.append(apatch)

        transformers = {}
        for m in range(M):
            transformers[m] = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1306,), (0.3081,)),
                 torchvision.transforms.Lambda((lambda y: lambda x: torch.max(x, patches[y]))(m)),
                ])
    elif perturbType == 'rotation':
        angles = np.arange(M) * 15 - 30
        transformers = {}
        for m in range(M):
      # load MNIST data
            transformers[m] = torchvision.transforms.Compose(
                [torchvision.transforms.RandomRotation((angles[m], angles[m])),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1306,), (0.3081,))
                ])
    elif perturbType == 'rotation30':
        angles = np.arange(M) * 30 - 60
        transformers = {}
        for m in range(M):
      # load MNIST data
            transformers[m] = torchvision.transforms.Compose(
                [torchvision.transforms.RandomRotation((angles[m], angles[m])),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1306,), (0.3081,))
                ])

    return transformers
  

def _lab_transformer_picker(mechChange, M):
    if not mechChange:
    # no hbar, no vbar
        return [None]*M
    else:
        # hbar is a label flipping feature
        res = [BarTransform('', 'h'),
               BarTransform('h', ''),
               BarTransform('', 'h'),
               BarTransform('h', ''),
               BarTransform('', 'h'),
               BarTransform('h', '')]
        for i in range(6, M):
            res.append(BarTransform('', 'h'))
        return res


def mnist_perturb_generic(perturbType='classic', M=2, 
                          interY=False, interYprop=0.5,
                          interYidx=[1],
                          mechChange=False, mechChangeProp=0.99,
                          datapath='../data',
                          batch_size=256):
    dataloaders = []
    testloaders = []

    transformers = _transformer_picker(perturbType, M)
    lab_dep_transforms = _lab_transformer_picker(mechChange, M)

    if mechChange:
    # for label dependent transform probability
        do_lab_transform_prob = mechChangeProp
    else:
        do_lab_transform_prob = 0
  
    for m in range(M):
        # transform the data
        dataset = MNISTlab(root=datapath, train=(m<M-1),
                            download=True, transform=transformers[m],
                            lab_dep_transform=lab_dep_transforms[m],
                            do_lab_transform_prob=do_lab_transform_prob)
    
        _subset_dataset(dataset, interY & (m in interYidx),
                        interYprop=interYprop, subsetProp=0.2 if (m<M-1) else 1.)

        dataloaders.append(torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, num_workers=0))

        source = dataloaders[:-1][::-1]
        target = dataloaders[-1]
        source.append(target)

    return source


def mnist_perturb_preset(perturb='rotation_1', M=2,
                         datapath='../data'):
  
    perturbType = perturb.split('_')[0]
#     if perturbType == 'classic':
#         return mnist_perturb_generic(perturbType=perturbType, M=M, 
#                               interY=False, 
#                               mechChange=False,
#                               datapath=datapath)
#     else:
#         
    daType = perturb.split('_')[1]
    if len(perturb.split('_')) == 2:
    
        if daType == '1':
            # da00: X shift, no Y shift, no mechanism shift
            return mnist_perturb_generic(perturbType=perturbType, M=M, 
                                  interY=False, 
                                  mechChange=False,
                                  datapath=datapath)
        elif daType == '2':
            # da01: X shift, Y shift, no mechanism shift  
            return mnist_perturb_generic(perturbType=perturbType, M=M, 
                              interY=True, interYprop=0.5,
                              interYidx=[M-1],
                              mechChange=False, mechChangeProp=0.9,
                              datapath=datapath)
        elif daType == '3':
            # da01: X shift, no Y shift, mechanism shift  
            return mnist_perturb_generic(perturbType=perturbType, M=M, 
                              interY=False,
                              mechChange=True, mechChangeProp=0.9,
                              datapath=datapath)
        elif daType == '4':
            # da01: X shift, Y shift, mechanism shift  
            return mnist_perturb_generic(perturbType=perturbType, M=M, 
                              interY=True, interYprop=0.5,
                              interYidx=[M-1],
                              mechChange=True, mechChangeProp=0.9,
                              datapath=datapath)
    else:
        if daType == '1':
            # da00: X shift, no Y shift, no mechanism shift
            dataloaders = mnist_perturb_generic(perturbType=perturbType, M=6, 
                                                interY=False, 
                                                mechChange=False,
                                                datapath=datapath)
            return [dataloaders[i] for i in [1,2,3,5]]
        elif daType == '2':
            # da01: X shift, Y shift, no mechanism shift  
            dataloaders = mnist_perturb_generic(perturbType=perturbType, M=6, 
                              interY=True, interYprop=0.5,
                              interYidx=[5],
                              mechChange=False, mechChangeProp=0.9,
                              datapath=datapath)
            return [dataloaders[i] for i in [1,2,3,5]]
        elif daType == '3':
            # da01: X shift, no Y shift, mechanism shift  
            dataloaders = mnist_perturb_generic(perturbType=perturbType, M=6, 
                              interY=False,
                              mechChange=True, mechChangeProp=0.9,
                              datapath=datapath)
            return [dataloaders[i] for i in [1,2,3,5]]
        elif daType == '4':
            # da01: X shift, Y shift, mechanism shift  
            dataloaders = mnist_perturb_generic(perturbType=perturbType, M=6, 
                              interY=True, interYprop=0.5,
                              interYidx=[5],
                              mechChange=True, mechChangeProp=0.9,
                              datapath=datapath)
            return [dataloaders[i] for i in [1,2,3,5]]
        
