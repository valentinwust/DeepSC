import torch
from torch.nn import Module
import numpy as np

##############################
##### Helper class for latent space evaluation
##############################

class EvaluateLatentModule():
    def __init__(self):
        super().__init__()
    
    def evaluate_latent(self, loader, device, training=False, **kwargs):
        """ Get latent representation for full dataloader.
        """
        with torch.no_grad():
            if not training is None:
                self.train(training)
            latents = []

            for batch in loader:
                k = batch[0].to(device)
                latents.append(self.get_latent(k, **kwargs).to("cpu").detach().numpy())
            latents = np.concatenate(latents)

            return latents
    
    def evaluate_means(self, loader, device, training=False, **kwargs):
        """ Get output mean expression for full dataloader.
        """
        with torch.no_grad():
            if not training is None:
                self.train(training)
            means = []

            for batch in loader:
                k = batch[0].to(device)
                means.append(self.forward(k, **kwargs)[0].to("cpu").detach().numpy())
            means = np.concatenate(means)

            return means