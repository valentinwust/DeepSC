from torch.nn import Module, Sequential, Linear
import numpy as np

from ..util import printwtime
from ..nn import RNA_PreprocessLayer, RNA_MeanActivation, RNA_DispersionActivation
from ..nn import make_FC_encoder, make_FC_decoder
from ..nn import NB_loss

from torch.distributions import Normal, kl_divergence

##############################
##### Simple NB Variational Autoencoder
##############################

class RNA_NBVariationalAutoEncoder(Module):
    """ Simple NB variational autoencoder, basically a simpler reimplementation of scVI.
        
        scVI learns a network for the size factor, this is completely pointless.
    """
    def __init__(self,
                 input_size,
                 output_size=None,
                 encoder_size=[64,32],
                 decoder_size=[64],
                 activation=True,
                 batchnorm=True,
                 dropout=0.,
                 bias=True,
                 BNmomentum=.9,
                 fixed_dispersion=None,
                 var_eps=1e-4,
                 kl_weight=1e-3,
                 kl_budget=2.):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        if self.output_size is None: self.output_size = input_size
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.batchnorm = batchnorm
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.var_eps = var_eps
        self.BNmomentum = BNmomentum
        self.fixed_dispersion = fixed_dispersion
        self.loss = NB_loss
        self.kl_weight = torch.tensor(kl_weight)
        self.kl_budget = torch.tensor(kl_budget)
    
    def build_network(self, counts, device=None):
        """ Build network, needs full counts to initialize preprocessing parameters.
        """
        self.pre = RNA_PreprocessLayer(self.input_size, counts)#.to(self.device))
        
        self.encoder = make_FC_encoder(self.input_size, self.encoder_size[:-1],
                                     batchnorm=self.batchnorm, activation=self.activation, dropout=self.dropout, bias=self.bias, BNmomentum=self.BNmomentum)
        
        self.encoder_mu = Linear(self.encoder_size[-2] if len(self.encoder_size)>1 else self.input_size, self.encoder_size[-1])
        self.encoder_var = Linear(self.encoder_size[-2] if len(self.encoder_size)>1 else self.input_size, self.encoder_size[-1])
        
        self.decoder = make_FC_decoder(self.encoder_size[-1], self.decoder_size,
                                     batchnorm=self.batchnorm, activation=self.activation, dropout=self.dropout, bias=self.bias, BNmomentum=self.BNmomentum)
        
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
        """ Forward pass through the network.
        """
        yc, s = self.pre(k)
        latent_intermediate = self.encoder(yc)
        latent_mu = self.encoder_mu(latent_intermediate)
        latent_var = torch.exp(self.encoder_var(latent_intermediate)) + self.var_eps
        latent = Normal(latent_mu, torch.sqrt(latent_var)).rsample()
        
        decoded = self.decoder(latent)
        rho = self.decoder_mu(decoded)
        mean = s * rho
        if self.fixed_dispersion is None:
            theta = self.decoder_theta(decoded)
        else:
            theta = torch.tensor(self.fixed_dispersion)
        return mean, theta, latent_mu, latent_var
    
    def get_latent(self, k, sample=True):
        """ Get latent representation.
            
            Probably better to sample here than to just return mean,
            since the most likely sample from a distribution isn't necessarily
            representative of most samples (https://benanne.github.io/2020/09/01/typicality.html).
        """
        yc, s = self.pre(k)
        latent_intermediate = self.encoder(yc)
        latent_mu = self.encoder_mu(latent_intermediate)
        if sample:
            latent_var = torch.exp(self.encoder_var(latent_intermediate)) + self.var_eps
            latent = Normal(latent_mu, torch.sqrt(latent_var)).rsample()
        else:
            latent = latent_mu
        return latent
    
    def get_loss(self, k, kout=None):
        """ Get loss for k, not mean reduced along batch.
        """
        mean, theta, latent_mu, latent_var = self.forward(k)
        loss = self.loss(k if kout is None else kout, mean, theta)
        
        kl_loss = self.kl_weight*kl_divergence(Normal(latent_mu, torch.sqrt(latent_var)),
                                               Normal(torch.zeros_like(latent_mu), torch.ones_like(latent_var))).mean(-1)
        return {"nll": loss, "kl_raw": kl_loss, "kl": torch.max(torch.tensor(0), kl_loss-self.kl_weight*self.kl_budget)}
    
    def evaluate_mean_loss(self, loader, device, training=False):
        """ Evaluate loss of model on whole data loader.
        """
        with torch.no_grad():
            if not training is None:
                self.train(training)
            total_loss = None
            length = len(loader.dataset)

            for batch in loader:
                kin = batch[0].to(device)
                kout = batch[1].to(device)
                loss = self.get_loss(kin, kout)
                if total_loss is None:
                    total_loss = {key:loss[key].sum().item()/length for key in loss}
                else:
                    for key in loss:
                        total_loss[key] += loss[key].sum().item()/length

            return total_loss
    
    def train_model(self, counts, batchsize=128, epochs=200, device="cuda:0", lr=1e-3, verbose=True, clip_gradients=1., kl_warmup=170, countsout=None):
        """ Train model.
        """
        trainloader, testloader = get_RNA_dataloaders([counts, counts if countsout is None else countsout], batch_size=batchsize)
        printwtime(f"Train model {type(self).__name__}")
        
        optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        printwtime(f"  Optimizer {type(optimizer).__name__} (lr={optimizer.param_groups[0]['lr']}), {epochs} epochs, device {device}.")
        
        history = {"training_loss":[], "test_loss":[], "epoch":[], "lr":[]}
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            weightkey = {"nll":1, "kl_raw":1, "kl":1 if epoch>=kl_warmup else epoch/kl_warmup}
            running_loss = None
            length = len(trainloader.dataset)
            self.train()
            for batch in trainloader:
                optimizer.zero_grad()

                datain = batch[0].to(device)
                dataout = batch[1].to(device)
                loss = self.get_loss(datain, dataout)
                loss = {key:loss[key].mean()*dataout.shape[0]*weightkey[key] for key in loss}
                (loss["nll"]+loss["kl"]).backward()
                if clip_gradients is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_gradients)
                optimizer.step()

                if running_loss is None:
                    running_loss = {key:loss[key].sum().item()/length for key in loss}
                else:
                    for key in loss:
                        running_loss[key] += loss[key].sum().item()/length
                #running_loss += loss.item()*data.shape[0]
            #running_loss = running_loss / 

            evalloss = self.evaluate_mean_loss(testloader, device, False)

            history["epoch"].append(epoch)
            history["training_loss"].append(running_loss)
            history["test_loss"].append(evalloss)

            if verbose: printwtime(f'  [{epoch + 1}/{epochs}] train loss: {running_loss["nll"]:.3f}, {running_loss["kl"]:.3f}, {running_loss["kl_raw"]:.3f}, test loss: {evalloss["nll"]:.3f}, {evalloss["kl"]:.3f}, {evalloss["kl_raw"]:.3f}')
            
        return history
    
    def evaluate_latent(self, loader, device, training=False, sample=True):
        """ Get latent representation for full dataloader.
        """
        with torch.no_grad():
            if not training is None:
                self.train(training)
            latents = []

            for batch in loader:
                k = batch[0].to(device)
                latents.append(self.get_latent(k, sample=sample).to("cpu").detach().numpy())
            latents = np.concatenate(latents)

            return latents

##############################
##### NB Total Correlation Variational Autoencoder, currently not functional!!!
##############################

class RNA_NBtcVariationalAutoEncoder(RNA_NBVariationalAutoEncoder):
    def __init__(self, *args, **kwargs, tc_weight=1):
        super(SubClass, self).__init__(*args, **kwargs)
        self.tc_weight = tc_weight
    
    def train_model(self, counts, batchsize=128, epochs=30, device="cuda:0", lr=1e-3, verbose=True, clip_gradients=1., kl_warmup=30):
        """ Train model.
        """
        trainloader, testloader = get_RNA_dataloaders(counts, batch_size=128)
        printwtime(f"Train model {type(self).__name__}")
        
        optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        printwtime(f"  Optimizer {type(optimizer).__name__} (lr={optimizer.param_groups[0]['lr']}), {epochs} epochs, device {device}.")
        
        history = {"training_loss":[], "test_loss":[], "epoch":[], "lr":[]}
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            weightkey = {"nll":1, "kl_raw":1, "kl":1 if epoch>kl_warmup else epoch/kl_warmup, "tc":1}
            running_loss = None
            length = len(trainloader.dataset)
            self.train()
            for batch in trainloader:
                optimizer.zero_grad()
                
                data = batch[0].to(device)
                loss = self.get_loss(data)
                loss = {key:loss[key].mean()*data.shape[0]*weightkey[key] for key in loss}
                (loss["nll"]+loss["kl"]).backward()
                if clip_gradients is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_gradients)
                optimizer.step()
                
                if running_loss is None:
                    running_loss = {key:loss[key].sum().item()/length for key in loss}
                else:
                    for key in loss:
                        running_loss[key] += loss[key].sum().item()/length
                #running_loss += loss.item()*data.shape[0]
            #running_loss = running_loss / 
            
            evalloss = self.evaluate_mean_loss(testloader, device, False)
            
            history["epoch"].append(epoch)
            history["training_loss"].append(running_loss)
            history["test_loss"].append(evalloss)
            
            if verbose: printwtime(f'  [{epoch + 1}/{epochs}] train loss: {running_loss["nll"]:.3f}, {running_loss["kl"]:.3f}, {running_loss["kl_raw"]:.3f}, {running_loss["tc"]:.3f}, test loss: {evalloss["nll"]:.3f}, {evalloss["kl"]:.3f}, {evalloss["kl_raw"]:.3f}, {evalloss["tc"]:.3f}')
            
        return history
    














