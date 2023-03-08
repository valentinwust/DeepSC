import torch
from torch.nn import Module
import numpy as np

##############################
##### Simple Embedding Layer
##############################

def normalize_embedding(embedding, dim=1, detach_norm=True, onlymax=True):
    """ Normalize embedding layer. Needs to detach norm for proper backpropagation.
    """
    norm = embedding.norm(dim=dim, keepdim=True)
    if detach_norm:
        norm = norm.detach()
    if onlymax:
        norm = torch.clip(norm, min=1)
    return embedding/norm

class EmbeddingContainer(Module):
    """ Simple embedding, always returns the full embedding.
        
        The default torch norm of embeddings seems to be only setting that max norm to 1,
        but properly trained it doesn't really fall below that anyway.
        Maybe just to avoid 1/0 problems?
    """
    def __init__(self, embedding_n, embedding_size=10, onlymax=True):
        super().__init__()
        self.embedding_n = embedding_n
        self.embedding_size = embedding_size
        self.onlymax = onlymax
        
        self.embedding = torch.nn.Parameter(torch.empty((embedding_n, embedding_size), dtype=torch.float32))
        torch.nn.init.normal_(self.embedding)
    
    def normed_embedding(self, onlymax=None):
        return normalize_embedding(self.embedding, dim=1, onlymax=self.onlymax if onlymax is None else onlymax)
    
    def forward(self, x):
        return x

##############################
##### Simple Embedding Layer
##############################

class RNA_EncodewGeneEmbeddingLayer(Module):
    """ Process RNA input, normalized, with gene embedding.
        
        Not sure about the initialization!
        Currently doesn't use separate scales for the internal dimension.
    """
    def __init__(self, output_size, embCont, internal_size=1, scale_in=True, scale_out=False, bias_in=True, bias_out=True):
        super().__init__()
        self.embCont = embCont
        self.input_size = self.embCont.embedding_n
        self.output_size = output_size
        self.embedding_size = self.embCont.embedding_size
        self.internal_size = internal_size
        self.use_scale_in = scale_in
        self.use_scale_out = scale_out
        self.use_bias_in = bias_in
        self.use_bias_out = bias_out
        
        self.weight = torch.nn.Parameter(torch.empty((self.embedding_size, self.internal_size, self.output_size), dtype=torch.float32))
        torch.nn.init.normal_(self.weight)
        
        if self.use_scale_in:
            self.scale_in = torch.nn.Parameter(torch.empty((1,self.input_size), dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.scale_in, a=np.sqrt(5))
        if self.use_scale_out:
            self.scale_out = torch.nn.Parameter(torch.empty((1,self.output_size), dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.scale_out, a=np.sqrt(5))
        
        if self.use_bias_in:
            self.bias_in = torch.nn.Parameter(torch.empty(self.input_size, dtype=torch.float32))
            torch.nn.init.normal_(self.bias_in)
        if self.use_bias_out:
            self.bias_out = torch.nn.Parameter(torch.empty(self.output_size, dtype=torch.float32))
            torch.nn.init.normal_(self.bias_out)
    
    def forward(self, x):
        out = x
        if self.use_scale_in:        out = out * self.scale_in
        if self.use_bias_in:         out = out + self.bias_in
        out = torch.einsum("ij, jk, klm -> im", out, self.embCont.normed_embedding(), normalize_embedding(self.weight, dim=1))
        if self.use_scale_out:        out = out * self.scale_out
        if self.use_bias_out:         out = out + self.bias_out
        return out

class RNA_DecodewGeneEmbeddingLayer(Module):
    """ Make RNA output with gene embedding.
        
        Use same initialization as Linear for scale/bias, is this the best?
        Currently doesn't use separate scales for the internal dimension.
    """
    def __init__(self, input_size, embCont, internal_size=1, scale_in=False, scale_out=True, bias=True):
        super().__init__()
        self.embCont = embCont
        self.input_size = input_size
        self.output_size = self.embCont.embedding_n
        self.embedding_size = self.embCont.embedding_size
        self.internal_size = internal_size
        self.use_scale_in = scale_in
        self.use_scale_out = scale_out
        self.use_bias = bias
        
        self.weight = torch.nn.Parameter(torch.empty((self.input_size, self.internal_size, self.embedding_size), dtype=torch.float32))
        torch.nn.init.normal_(self.weight)
        
        if self.use_scale_in:
            self.scale_in = torch.nn.Parameter(torch.empty((1,self.input_size), dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.scale_in, a=np.sqrt(5))
        if self.use_scale_out:
            self.scale_out = torch.nn.Parameter(torch.empty((1,self.output_size), dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.scale_out, a=np.sqrt(5))
        
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.empty(self.output_size, dtype=torch.float32))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        out = x
        if self.use_scale_in:   out = out * self.scale_in
        out = torch.einsum("ij, lm, jkm -> il", out, self.embCont.normed_embedding(), normalize_embedding(self.weight, dim=1))
        if self.use_scale_out:  out = out * self.scale_out
        if self.use_bias:       out = out + self.bias
        return out

