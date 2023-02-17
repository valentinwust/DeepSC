from torch.nn import Module, Sequential, Linear
import torch

from ..util import printwtime
from ..nn import RNA_PreprocessLayer, RNA_MeanActivation, RNA_DispersionActivation
from ..nn import make_FC_encoder, make_FC_decoder
from ..nn import NB_loss
from ..util import get_RNA_dataloaders

class RNA_NBAutoEncoder(Module):
    """ Simple NB autoencoder, basically reimplementation of dca.
        
        Can use a fixed dispersion.
        BatchNorm seems to train fine now, but why isn't there the same difference between train/test loss
        as for dca in keras?????
    """
    def __init__(self,
                 input_size,
                 output_size=None,
                 encoder_size=[64,32],
                 decoder_size=[64],
                 activation=True,
                 batchnorm=True,
                 bias=True,
                 BNmomentum=.9,
                 fixed_dispersion=None):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        if self.output_size is None: self.output_size = input_size
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.batchnorm = batchnorm
        self.activation = activation
        self.bias = bias
        self.BNmomentum = BNmomentum
        self.fixed_dispersion = fixed_dispersion
        self.loss = NB_loss
    
    def build_network(self, counts, device=None):
        """ Build network, needs full counts to initialize preprocessing parameters.
        """
        self.pre = RNA_PreprocessLayer(self.input_size, counts)#.to(self.device))
        
        self.encoder = make_FC_encoder(self.input_size, self.encoder_size,
                                     batchnorm=self.batchnorm, activation=self.activation, bias=self.bias, BNmomentum=self.BNmomentum)
        
        self.decoder = make_FC_decoder(self.encoder_size[-1], self.decoder_size,
                                     batchnorm=self.batchnorm, activation=self.activation, bias=self.bias, BNmomentum=self.BNmomentum)
        
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
         
    def forward(self, k):
        yc, s = self.pre(k)
        latent = self.encoder(yc)
        decoded = self.decoder(latent)
        rho = self.decoder_mu(decoded)
        mean = s * rho
        if self.fixed_dispersion is None:
            theta = self.decoder_theta(decoded)
        else:
            theta = torch.tensor(self.fixed_dispersion)
        return mean, theta
    
    def get_latent(self, k, wBNA=True):
        """ Get latent representation, optionally without BN and activation function.
        """
        yc, s = self.pre(k)
        latent = self.encoder(yc) if wBNA else self.encoder[:-sum([self.activation+self.batchnorm])](yc)
        return latent
    
    def get_loss(self, k):
        """ Get loss for k, not mean reduced along batch.
        """
        mean, theta = self.forward(k)
        loss = self.loss(k, mean, theta)
        return {"nll": loss}
    
    def evaluate_mean_loss(self, loader, device, training=False):
        """ Evaluate loss of model on whole data loader.
        """
        with torch.no_grad():
            if not training is None:
                self.train(training)
            total_loss = None

            for batch in loader:
                k = batch[0].to(device)
                total_loss += self.get_loss(k)["nll"].sum().item()
        
            return total_loss/len(loader.dataset)
    
    def train(self, counts, batchsize=128, epochs=30, device="cuda:0", lr=1e-3, verbose=True):
        """ Train model.
        """
        trainloader, testloader = get_RNA_dataloaders(counts, batch_size=128)
        printwtime(f"Train model {type(self).__name__}")
        
        optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        printwtime(f"  Optimizer {type(optimizer).__name__} (lr={optimizer.param_groups[0]['lr']}), {epochs} epochs, device {device}.")
        
        history = {"training_loss":[], "test_loss":[], "epoch":[], "lr":[]}
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            self.train()
            for batch in trainloader:
                optimizer.zero_grad()
                
                data = batch[0].to(device)
                loss = self.get_loss(data)["nll"].mean()
                loss.backward()
                if clip_gradients is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_gradients)
                optimizer.step()
                
                running_loss += loss.item()*data.shape[0]
            running_loss = running_loss / len(trainloader.dataset)
            
            evalloss = self.evaluate_mean_loss(testloader, device, False)
            
            history["epoch"].append(epoch)
            history["training_loss"].append(running_loss)
            history["test_loss"].append(evalloss)
            
            if verbose: printwtime(f'  [{epoch + 1}/{epochs}] train loss: {running_loss:.3f}, test loss: {evalloss:.3f}')
            
        return history

class RNA_NBPCA(RNA_NBAutoEncoder):
    """ Simplified version of RNA_NBAutoEncoder, similar to glm-pca?
    """
    def __init__(self,
                 input_size,
                 dispersion,
                 latent=32):
        
        super().__init__(input_size=input_size,
                 output_size=None,
                 encoder_size=[latent],
                 decoder_size=[],
                 activation=False,
                 batchnorm=False,
                 bias=True,
                 fixed_dispersion=dispersion)
