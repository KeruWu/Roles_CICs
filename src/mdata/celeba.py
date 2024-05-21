
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

import torch
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader,Dataset


def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)

def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)



class ColorEnhance:
    def __init__(self, v):
        self.v = v
    def __call__(self, img):
        return Color(img, self.v)

class BrightnessEnhance:
    def __init__(self, v):
        self.v = v
    def __call__(self, img):
        return Brightness(img, self.v)
    
class ContrastEnhance:
    def __init__(self, v):
        self.v = v
    def __call__(self, img):
        return Contrast(img, self.v)    
    

class CelebA_Dataset(Dataset):
    def __init__(self, image_path, df_attr, idxs, label='Smiling', transform=None, transform2=None, prob=1):
        self.image_path = image_path
        self.attr = df_attr
        self.idxs = idxs
        self.transform = transform
        self.transform2 = transform2
        self.label = label
        self.prob = prob
        
    def __getitem__(self, index):
        image_name = self.idxs[index]
        x = Image.open(f'{self.image_path}/{image_name}')
        y = self.attr[self.label].loc[self.idxs[index]]
        y = 1 if y > 0 else 0
        if self.transform is not None:
            if y == 1:
                if np.random.uniform(size=1) < self.prob:
                    x = self.transform(x)
                else:
                    x = self.transform2(x)
            else:
                if np.random.uniform(size=1) < self.prob:
                    x = self.transform2(x)
                else:
                    x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.idxs)



def celeba_perturb(perturb, image_path, attr_path, n_domains=4, y='Smiling', spu='Mouth_Slightly_Open',
                   spu_size = [[2000, 8000, 2000, 8000],
                               [8000, 2000, 8000, 2000],
                               [9000, 1000, 9000, 1000],
                               [1000, 9000, 1000, 9000]],
                   color_enhance = [[1.,1.,1.,1.],
                                    [0.,0.,0.,0.,]],
                   prob = [0.9,0.6,0.3,0.1],
                   batch_size = 64):
    
    df_attr = pd.read_csv(attr_path, delimiter='\s+')
    
    idx = np.random.permutation(df_attr.shape[0])
    n = df_attr.shape[0]//n_domains
    dataloaders = []
    
    if int(perturb[-1]) <= 3:

        if perturb == 'CelebA_1':


            spu_size = [[9500, 500, 500, 9500],
                                   [9500, 500, 500, 9500],
                                   [500, 9500, 9500, 500],
                                   [500, 9500, 9500, 500]]

            color_enhance = [[1.,1.,1.,1.],
                             [0.,0.,0.,0.,]]
            prob = [0.8,0.2,0.1,0.9]

        elif perturb == 'CelebA_2':

            spu_size = [[7500, 2500, 2500, 7500],
                                   [7500, 2500, 2500, 7500],
                                   [2500, 7500, 7500, 2500],
                                   [2500, 7500, 7500, 2500]]

            color_enhance = [[1.,1.,1.,1.],
                             [0.,0.,0.,0.,]]
            prob = [0.8,0.2,0.1,0.9]

        elif perturb == 'CelebA_3':


            spu_size = [[9500, 500, 500, 9500],
                                   [9500, 500, 500, 9500],
                                   [500, 9500, 9500, 500],
                                   [500, 9500, 9500, 500]]

            color_enhance = [[1.,1.,1.,1.],
                             [0.,0.,0.,0.,]]
            prob = [0.8,0.7,0.6,0.9]




        df_y1_spu1 = df_attr[(df_attr[y]==1) & (df_attr[spu]==1)].sample(n=20000)
        df_y1_spu0 = df_attr[(df_attr[y]==1) & (df_attr[spu]==-1)].sample(n=20000)
        df_y0_spu1 = df_attr[(df_attr[y]==-1) & (df_attr[spu]==1)].sample(n=20000)
        df_y0_spu0 = df_attr[(df_attr[y]==-1) & (df_attr[spu]==-1)].sample(n=20000)

        dfs = [df_y1_spu1, df_y1_spu0, df_y0_spu1, df_y0_spu0]
        count = [0, 0, 0, 0]

        for i in range(n_domains):
            df_i = []
            for j in range(4):
                df_i.append(dfs[j].iloc[count[j]:(count[j]+spu_size[i][j]),:])
                count[j] += spu_size[i][j]
            df = pd.concat(df_i)

            dat = CelebA_Dataset(image_path = image_path,
                                 df_attr = df_attr,
                                 idxs = df.index,
                                 label = y,
                                 transform = transforms.Compose([
                                     transforms.Resize((64, 48)),
                                     ColorEnhance(v=color_enhance[0][i]),
                                     transforms.ToTensor(),
                                 ]),
                                 transform2 = transforms.Compose([
                                     transforms.Resize((64, 48)),
                                     ColorEnhance(v=color_enhance[1][i]),
                                     transforms.ToTensor(),
                                 ]),
                                 prob=prob[i])
            dataloaders.append(DataLoader(dat, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))

    else:
        
        if perturb == 'CelebA_4':

            spu_size = [[10000, 10000],
                        [10000, 10000],
                        [10000, 10000],
                        [10000, 3333]]

            color_enhance = [[1.,1.,1.,1.],
                             [0.,0.,0.,0.,]]
            prob = [0.8,0.2,0.1,0.9]


        elif perturb == 'CelebA_5':

            spu_size = [[10000, 10000],
                        [10000, 10000],
                        [10000, 10000],
                        [3333, 10000]]

            color_enhance = [[1.,1.,1.,1.],
                             [0.,0.,0.,0.,]]
            prob = [0.8,0.2,0.1,0.9]
            

        elif perturb == 'CelebA_6':

            spu_size = [[10000, 10000],
                        [10000, 10000],
                        [10000, 10000],
                        [10000, 3333]]

            color_enhance = [[1.,1.,1.,1.],
                             [0.,0.,0.,0.,]]
            prob = [0.8,0.7,0.6,0.9]


        

        df_y1 = df_attr[(df_attr[y]==1)].sample(n=40000)
        df_y0 = df_attr[(df_attr[y]==-1)].sample(n=40000)

        dfs = [df_y1, df_y0]
        count = [0, 0, 0, 0]

        for i in range(n_domains):
            df_i = []
            for j in range(2):
                df_i.append(dfs[j].iloc[count[j]:(count[j]+spu_size[i][j]),:])
                count[j] += spu_size[i][j]
            df = pd.concat(df_i)

            dat = CelebA_Dataset(image_path = image_path,
                                 df_attr = df_attr,
                                 idxs = df.index,
                                 label = y,
                                 transform = transforms.Compose([
                                     transforms.Resize((64, 48)),
                                     ColorEnhance(v=color_enhance[0][i]),
                                     transforms.ToTensor(),
                                 ]),
                                 transform2 = transforms.Compose([
                                     transforms.Resize((64, 48)),
                                     ColorEnhance(v=color_enhance[1][i]),
                                     transforms.ToTensor(),
                                 ]),
                                 prob=prob[i])
            dataloaders.append(DataLoader(dat, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))


    
    return dataloaders

