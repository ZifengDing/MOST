import numpy as np
# PyTorch related imports
import torch
from torch.nn.init import xavier_normal_
from torch.nn import Parameter

np.set_printoptions(precision=4)

def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param

