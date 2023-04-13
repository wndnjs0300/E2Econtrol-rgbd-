import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils_2d import PointNetSetAbstraction#, PointNetSetAbstractionMsg 
from torchvision import models
import timm

from einops.layers.torch import Rearrange
from transformer import Transformer

print(F.one_hot(torch.arange(15), num_classes=15))