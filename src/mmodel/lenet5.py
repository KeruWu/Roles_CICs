import torch.nn as nn
import torch.nn.functional as F

from .layers import Identity

## LeNet5 CNN model for MNIST

class LeNet5(nn.Module):
    def __init__(self, nb_classes=2):
        super().__init__()
        self.featurizer = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(4*4*50, 500),
            nn.ReLU(),
            nn.Linear(500, nb_classes)
        )
        self.classifier = Identity(nb_classes)

    def forward(self, x):
        return self.featurizer(x)

    
## CNN model for CelebA
    
class CNN_CelebA(nn.Module):
    def __init__(self, nb_classes=2):
        super().__init__()
        self.featurizer = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1),  
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # 30, 22
            nn.Conv2d(16, 32, 5, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # 14, 10 
            nn.Conv2d(32, 64, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # 5,  3 
            nn.Flatten(),
            nn.Linear(5*3*64, 256),
            nn.ReLU(),
            nn.Linear(256, nb_classes)
        )
        self.classifier = Identity(nb_classes)

    def forward(self, x):
        return self.classifier(self.featurizer(x))
    