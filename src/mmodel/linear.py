import torch.nn as nn
from .layers import Identity


class Linear(nn.Module):
    """Linear model takes d dim input and output nb_classes
    """
    def __init__(self, d, nb_classes=2):
        super().__init__()
        self.nb_classes = nb_classes
        self.featurizer = nn.Linear(d, nb_classes, bias=True)
        self.classifier = Identity(nb_classes)
        # custom initialization
        nn.init.xavier_normal_(self.featurizer.weight, gain=0.01)

    def forward(self, x):
        x = self.featurizer(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_nodes_list, n_classes):
        super(MLP, self).__init__()

        modules = []
        dimensions = [input_dim] + hidden_nodes_list

        for i in range(len(dimensions)-1):
            modules.append(nn.Linear(dimensions[i], dimensions[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(dimensions[-1], n_classes))

        self.featurizer = nn.Sequential(*modules)
        self.classifier = Identity(n_classes)

    def forward(self, x):
        x = self.featurizer(x)
        return x