import torch
from torch.nn import Module
import numpy as np

# Import shap here and add custom layers
import shap
from shap.explainers._deep.deep_pytorch import nonlinear_1d
shap.explainers._deep.deep_pytorch.op_handler["RNA_MeanActivation"] = nonlinear_1d
shap.explainers._deep.deep_pytorch.op_handler["RNA_DispersionActivation"] = nonlinear_1d

""" Set all used values as Parameter, then the model shifts them to the right device automatically!
    
"""

##############################
##### Preprocessing
##############################

class RNA_PreprocessLayer(Module):
    """ RNA count preprocessing layer.
        
        Mean and offset are optionally trainable.
        For simple NB AE and NB PCA, this does not seem to make any difference, and it barely changes them.
        
        Initial offset is assumed to be the exponent in base 10.
    """
    def __init__(self, N, counts, means_trainable=False, offset_trainable=False, initial_offset=-4.):
        super().__init__()
        self.means = torch.nn.Parameter(torch.log(counts/counts.sum(dim=-1)[:,None] + torch.exp(torch.tensor(initial_offset))).mean(0))
        self.means.requires_grad = means_trainable
        self.offset = torch.nn.Parameter(torch.tensor([initial_offset*np.log(10.)], dtype=torch.float).repeat(N))
        self.offset.requires_grad = offset_trainable
    
    def forward(self, k):
        s = k.sum(dim=-1, keepdim=True)
        y = torch.log(k / s + torch.exp(self.offset))
        return y-self.means, s
    
    def normalize_counts(self, k):
        yc, s = self.forward(k)
        return yc

##############################
##### Output Activations
##############################

class RNA_MeanActivation(Module):
    """ Softmax activation function.
    """
    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=-1)
        #return torch.clip(torch.exp(x), min=1e-5, max=1e6)

class RNA_DispersionActivation(Module):
    """ Clipped softplus activation function.
    """
    def forward(self, x):
        return torch.clip(torch.exp(x), min=1e-4, max=1e6)
        #return torch.clip(torch.nn.functional.softplus(x), min=1e-4, max=1e4)