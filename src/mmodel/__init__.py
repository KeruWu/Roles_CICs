"""
mmodel implements supported model choices
"""
from .layers import Identity
from .linear import Linear, MLP
from .lenet5 import LeNet5, CNN_CelebA
from .densenet import DenseNet
from .resnet import ResNet