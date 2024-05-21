import torch.nn as nn
import torch.nn.functional as F

from .layers import Identity
from torchvision.models import resnet101

## ResNet152 with flattened CNN output as featurizer. 
    
class ResNet(nn.Module):
    def __init__(self, nb_classes=2):
        super().__init__()
        self.nb_classes = nb_classes
        self.featurizer = resnet101(pretrained=True)  #.features # remove fc layer
        self.d_out = self.featurizer.fc.in_features
        self.featurizer.fc = Identity(self.d_out)
        self.classifier = nn.Linear(self.d_out, self.nb_classes)
    
    def forward(self, x):
        feats = self.featurizer(x)
        out = self.classifier(feats)
        return out
    
    