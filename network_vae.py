import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from default_from_cfg import DefaultFromCfg
import cfg

class NetworkVAE(nn.Module) :
    
    def __init__(self,
                 Nlatent=DefaultFromCfg('VAE_NLATENT'),
                 rand_latent=DefaultFromCfg('VAE_RAND_LATENT'),
                 **MLP_kwargs) :
    #{{{
        if isinstance(Nlatent, DefaultFromCfg) :
            Nlatent = Nlatent()
        if isinstance(rand_latent, DefaultFromCfg) :
            rand_latent = rand_latent()

        super().__init__()

        if not rand_latent :
            self.mlp = NetworkMLP(cfg.RESIDUALS_NBINS, 2*Nlatent, **MLP_kwargs)

        self.Nlatent = Nlatent
        self.rand_latent = rand_latent
    #}}}
    
    def forward(self, x) :
        """
        x ... the input, of shape [batch, Nbins]. 
              Must also be passed when rand_latent is True as this is a convenient way
              to get the batch size and device

        Returns the encoded latent space representation and the KL loss (None if rand_latent is True)
        where the KL loss is not summed over batch, i.e. a [batch] tensor
        """
    #{{{ 
        if self.rand_latent :
            # draw random latent space variables
            return torch.randn(x.shape[0], Nlatent, device=x.device), None

        # pass through the network
        h = self.mlp(x)
        mu = h[:, :self.Nlatent]
        logvar = h[:, self.Nlatent:]

        # compute the latent space variables
        z = mu + torch.exp(0.5*logvar) * torch.randn(x.shape[0], self.Nlatent, device=x.device)

        # compute negative KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1) 

        return z, KLD
    #}}}
