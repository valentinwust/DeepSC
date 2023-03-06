import torch
from torch.nn import Module
import numpy as np

""" Set all used values as Parameter, then the model shifts them to the right device automatically!
    
    SHAP cannot deal with non-linearities that are not explicitly in a torch layer!
    E.g. log1p has to be put in its own layer to get the right results.
"""

# Import shap here and add custom layers
import shap
from shap.explainers._deep.deep_pytorch import nonlinear_1d
shap.explainers._deep.deep_pytorch.op_handler["RNA_MeanActivation"] = nonlinear_1d
shap.explainers._deep.deep_pytorch.op_handler["RNA_DispersionActivation"] = nonlinear_1d
shap.explainers._deep.deep_pytorch.op_handler["RNA_Log1pActivation"] = nonlinear_1d

##############################
##### Preprocessing
##############################

class RNA_PreprocessLayer(Module):
    """ RNA count preprocessing layer.
        
        Scale needs some safeguards!!!!! Otherwise /0 and problems!!!
        
        Mean, std and offset are optionally trainable.
        For simple NB AE and NB PCA, this does not seem to make any difference, and it barely changes them.
        
        Initial offset is assumed to be the exponent in base 10.
    """
    def __init__(self, N, counts=None, shift=True, scale=False, means_trainable=False, stds_trainable=False, offset_trainable=False, initial_offset=-4.):
        super().__init__()
        self.shift = shift
        self.scale = scale
        if counts is not None and self.shift:
            vals = torch.log(counts/counts.sum(dim=-1)[:,None] + torch.exp(torch.tensor(initial_offset)))
            if self.shift:
                self.means = torch.nn.Parameter(vals.mean(0))
                self.means.requires_grad = means_trainable
                if self.scale:
                    self.stds = torch.nn.Parameter(vals.std(0))
                    self.stds.requires_grad = stds_trainable
        self.offset = torch.nn.Parameter(torch.tensor([initial_offset*np.log(10.)], dtype=torch.float).repeat(N))
        self.offset.requires_grad = offset_trainable
    
    def forward(self, k):
        s = k.sum(dim=-1, keepdim=True)
        y = torch.log(k / s + torch.exp(self.offset))
        if self.shift:
            y = y-self.means
            if self.scale:
                y = y/self.stds
        return y, s
    
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

class RNA_Log1pActivation(Module):
    """ Log(x+1).
    """
    def forward(self, x):
        return torch.log(x+1)


