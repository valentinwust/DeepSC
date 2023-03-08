from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU
import torch
import numpy as np

from ..util import printwtime
from ..nn import RNA_PreprocessLayer, RNA_MeanActivation, RNA_DispersionActivation, RNA_Log1pActivation
from ..nn import make_FC_encoder, make_FC_decoder
#from ..nn import NB_loss
#from ..util import get_RNA_dataloaders, get_RNA_dataloader
#from ..util import sample_indices
#from ..module import EvaluateLatentModule

from ..model import RNA_NBAutoEncoder
from ..nn import EmbeddingContainer, normalize_embedding



class RNA_GeneEmbeddingLayer(Module):
    """ 
    """
    def __init__(self, embCont, use_offset=False):
        super().__init__()
        self.embCont = embCont
        
        self.input_size = self.embCont.embedding_n
        self.embedding_size = self.embCont.embedding_size
        self.use_offset = use_offset
        
        self.scale = torch.nn.Parameter(torch.empty((self.input_size,1), dtype=torch.float32))
        torch.nn.init.normal_(self.scale)
        
        if self.use_offset:
            self.offset = torch.nn.Parameter(torch.empty(self.input_size, dtype=torch.float32))
            torch.nn.init.normal_(self.offset)
    
    def forward(self, x):
        xof = x if not self.use_offset else x + self.offset
        return xof[...,None] * self.embCont.normed_embedding()
        #return xof[...,None] * torch.nn.functional.normalize(self.embedding, p=2.0, dim=1) * torch.exp(self.scale)
    
    def get_normalized_embedding(self, onlymax=True):
        return self.embCont.embedding(onlymax).detach().to("cpu").numpy()
        #return self.embedding/(self.embedding**2).sum(axis=1)[:,None]

class RNA_ProcessGeneEmbeddingLayer(Module):
    """ 
    """
    def __init__(self, output_size, embedding_size=10, internal_size=1):
        super().__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.internal_size = internal_size
        
        self.weight = torch.nn.Parameter(torch.empty((embedding_size, internal_size, output_size), dtype=torch.float32))
        torch.nn.init.normal_(self.weight)
        
        self.scale = torch.nn.Parameter(torch.empty((internal_size, output_size), dtype=torch.float32))
        torch.nn.init.normal_(self.scale)
        
        self.bias = torch.nn.Parameter(torch.empty(output_size, dtype=torch.float32))
        
        # Use the same initialization as Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        #return (x @ self.weight).sum(axis=-3).sum(axis=-2) + self.bias
        return torch.einsum("ijk, klm -> im", x, normalize_embedding(self.weight, dim=0)) + self.bias
        #return torch.einsum("ijk, lm, klm -> im", x, self.scale, self.weight / self.weight.norm(dim=0, keepdim=True).detach()) + self.bias
        #return torch.einsum("ijk, klm -> im", x, torch.nn.functional.normalize(self.weight, p=2.0, dim=0)) + self.bias



##############################
##### Simple NB Autoencoder
##############################

class RNA_NBEmbeddingAutoEncoder(RNA_NBAutoEncoder):
    """ 
    """
    
    def __init__(self,
                 input_size,
                 embedding_size=10,
                 embedding_in=True,
                 embedding_out=True,
                 embedding_in_kwargs={},
                 embedding_out_kwargs={}
                 **kwargs):
        super().__init__(input_size, **kwargs)
        
        self.embedding_size = embedding_size
        self.embedding_in = embedding_in
        self.embedding_out = embedding_out
        self.embedding_in_kwargs = embedding_in_kwargs
        self.embedding_out_kwargs = embedding_out_kwargs
        
        if self.fixed_dispersion is None:
            raise ValueError("Currently only works with fixed dispersion!")
    
    def build_network(self, counts, device=None):
        """ Build network, needs full counts to initialize preprocessing parameters.
        """
        self.pre = RNA_PreprocessLayer(self.input_size, counts, means_trainable=True, shift=False)
        
        self.embCont = EmbeddingContainer(self.input_size, embedding_size=self.embedding_size)
        
        if self.embedding_in:
            self.encoder = Sequential(  RNA_EncodewGeneEmbeddingLayer(self.encoder_size[0], self.embCont, **self.embedding_in_kwargs),
                                        #self.embCont, # Does nothing, only here for proper overview
                                        #RNA_GeneEmbeddingLayer(self.embCont, use_offset=False),
                                        #RNA_ProcessGeneEmbeddingLayer(self.encoder_size[0], embedding_size=self.embedding_size),
                                        ReLU(),
                                        BatchNorm1d(self.encoder_size[0], momentum=self.BNmomentum),
                                        make_FC_encoder(self.encoder_size[0], self.encoder_size[1:],
                                             batchnorm=self.batchnorm, activation=self.activation, dropout=self.dropout, bias=self.bias, BNmomentum=self.BNmomentum,
                                             final_activation=self.latent_activation))
        else:
            self.encoder = make_FC_encoder(self.input_size, self.encoder_size,
                                             batchnorm=self.batchnorm, activation=self.activation, dropout=self.dropout, bias=self.bias, BNmomentum=self.BNmomentum,
                                             final_activation=self.latent_activation)
        
        self.decoder = make_FC_decoder(self.encoder_size[-1], self.decoder_size,
                                     batchnorm=self.batchnorm, activation=self.activation, dropout=self.dropout, bias=self.bias, BNmomentum=self.BNmomentum)
        
        if self.embedding_out:
            self.decoder_mu = Sequential(
                                            RNA_DecodewGeneEmbeddingLayer(self.decoder_size[-1], self.embCont, **self.embedding_out_kwargs),
                                            RNA_MeanActivation())
        else:
            self.decoder_mu = Sequential(
                                            Linear(self.decoder_size[-1] if len(self.decoder_size)>0 else self.encoder_size[-1],
                                                   self.output_size, bias=self.bias),
                                            RNA_MeanActivation())
        
        if self.fixed_dispersion is None:
            self.decoder_theta = Sequential(
                                        Linear(self.decoder_size[-1] if len(self.decoder_size)>0 else self.encoder_size[-1],
                                               self.output_size, bias=self.bias),
                                        RNA_DispersionActivation())
        
        if device is not None:
            self.to(device)


