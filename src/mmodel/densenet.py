import torch.nn as nn
import torch.nn.functional as F

from .layers import Identity
from torchvision.models import densenet121

## DenseNet121 with flattened CNN output as featurizer. 
    
class DenseNet(nn.Module):
    def __init__(self, nb_classes=2):
        super().__init__()
        self.nb_classes = nb_classes
        self.featurizer = densenet121(pretrained=False)  #.features # remove fc layer
        self.d_out = self.featurizer.classifier.in_features
        self.featurizer.classifier = Identity(self.d_out)
        self.classifier = nn.Linear(self.d_out, self.nb_classes)
    
    def forward(self, x):
        feats = self.featurizer(x)
        out = self.classifier(feats)
        return out
    
    