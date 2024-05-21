
import copy
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSSubset

from .randaugment import FIX_MATCH_AUGMENTATION_POOL, RandAugment

    
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]
    
    
def add_rand_augment_transform(base_transform_steps, normalization, randaugment_n=2):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = (96, 96)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution
            ),
            RandAugment(
                n=randaugment_n,
                augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return transforms.Compose(strong_transform_steps)

def get_center_subset(dataset, center=0, transform=None):
    split_mask = np.logical_and(dataset._metadata_df.center == center, dataset._metadata_df.split == 0)
    split_idx = np.where(split_mask)[0]
    return WILDSSubset(dataset, split_idx, transform)

def load_camelyon(root_dir, download=False, random_augment=False):
    dataset = get_dataset(dataset='camelyon17', download=download, root_dir=root_dir)
    unlabeled_dataset = get_dataset(dataset='camelyon17', download=download, root_dir=root_dir, unlabeled=True)
    
    target_resolution = (96, 96)
    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    transform_steps = []
    transform_steps.append(transforms.Resize(target_resolution))
    transform_steps.append(transforms.ToTensor())
    transform_steps.append(default_normalization)
    transform = transforms.Compose(transform_steps)
    
    transform_steps_aug = []
    transform_steps_aug.append(transforms.Resize(target_resolution))
    transform_aug = add_rand_augment_transform(transform_steps_aug, default_normalization)
    
    if random_augment:
        final_transform = transform_aug
    else:
        final_transform = transform
    
    train_data = dataset.get_subset("train", transform=final_transform)
    
    target_data = unlabeled_dataset.get_subset("test_unlabeled", transform=final_transform)
    target_data_labeled = dataset.get_subset("test", transform=final_transform)
    
        
    id_val_data = dataset.get_subset("id_val", transform=transform)
    val_data = dataset.get_subset("val", transform=transform)
    test_data = dataset.get_subset("test", transform=transform)
    grouper = CombinatorialGrouper(dataset, ['hospital'])

    train_loader = get_train_loader("standard", train_data, batch_size=32, num_workers=4, pin_memory=True)
    train_loader_grouped = get_train_loader("group", train_data, grouper=grouper, n_groups_per_batch=2, 
                                            batch_size=32, num_workers=4, pin_memory=True)
        
    target_loader = get_train_loader("standard", target_data, batch_size=64, num_workers=4, pin_memory=True)
    target_loader_labeled = get_train_loader("standard", target_data_labeled, batch_size=32, num_workers=4, pin_memory=True)
    
    
    id_val_loader = get_eval_loader("standard", id_val_data, batch_size=1024, num_workers=4, pin_memory=True)
    val_loader = get_eval_loader("standard", val_data, batch_size=1024, num_workers=4, pin_memory=True)
    test_loader = get_eval_loader("standard", test_data, batch_size=1024, num_workers=4, pin_memory=True)
    
    return train_loader, train_loader_grouped, target_loader, target_loader_labeled, id_val_loader, val_loader, test_loader, grouper

    