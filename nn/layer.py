import torch
from torch.nn import Module

""" Set all used values as Parameter, then the model shifts them to the right device automatically!
    
"""

##############################
##### Preprocessing
##############################

class RNA_PreprocessLayer(Module):
    """ RNA count preprocessing layer.
        
        Mean and offset are optionally trainable.
        For simple NB AE and NB PCA, this does not seem to make any difference, and it barely changes them.
    """
    def __init__(self, N, counts, means_trainable=False, offset_trainable=False, initial_offset=-4.):
        super().__init__()
        self.means = torch.nn.Parameter(torch.log(counts/counts.sum(dim=-1)[:,None] + torch.exp(torch.tensor(initial_offset))).mean(0))
        self.offset = torch.nn.Parameter(torch.tensor([initial_offset]).repeat(N))
        if means_trainable:
            self.means.requires_grad = True
        if offset_trainable:
            self.offset.requires_grad = True
    
    def forward(self, k):
        s = k.sum(dim=-1, keepdim=True)
        y = torch.log(k / s + torch.exp(self.offset))
        return y-self.means, s 

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