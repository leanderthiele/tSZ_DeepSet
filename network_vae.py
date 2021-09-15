import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from default_from_cfg import DefaultFromCfg
import cfg

class NetworkVAE(nn.Module) :
    
    def __init__(self,
                 Nlatent=DefaultFromCfg('VAE_NLATENT'),
                 **MLP_kwargs) :
    #{{{
        if isinstance(Nlatent, DefaultFromCfg) :
            Nlatent = Nlatent()

        super().__init__()

        self.mlp = NetworkMLP(cfg.RESIDUALS_NBINS, 2*Nlatent, **MLP_kwargs)

        self.Nlatent = Nlatent

        # we initialize this to None so that when the network is put in DDP mode
        # not the same RNG is copied to all the processes
        # We simply initialize the RNG on the first forward call
        self.rng = None
    #}}}


    def create_scalars(self, x) :
    #{{{ 
        desc = 'input to NetworkVAE: x [%d]; '%cfg.RESIDUALS_NBINS
        return x, desc
    #}}}
    

    def forward(self, x, gaussian, seed=None) :
        """
        x ... the input, of shape [batch, Nbins]. 
              Must also be passed when gaussian is True as this is a convenient way
              to get the batch size and device
        gaussian ... whether the output is a pure N(0,1) number or sampled from the encoded distribution
        seed ... integer to manually initialize the RNG (otherwise the internal RNG will be used)

        Returns the encoded latent space representation and the KL loss (only if gaussian=False)
        where the KL loss is not summed over batch, i.e. a [batch] tensor
        """
    #{{{ 
        if self.rng is None :
            # seed with different number for each process
            self.rng = torch.Generator(device=x.device).manual_seed(int((cfg.NETWORK_SEED+cfg.rank) % 2**63))

        if seed is not None :
            _rng = torch.Generator(device=x.device).manual_seed(int(seed))
        else :
            _rng = self.rng

        if gaussian :
            # draw random latent space variables
            return torch.randn(x.shape[0], self.Nlatent, device=x.device, generator=_rng)

        scalars, _ = self.create_scalars(x)

        # pass through the network
        h = self.mlp(scalars)
        mu = h[:, :self.Nlatent]
        logvar = h[:, self.Nlatent:]

        # compute the latent space variables
        z = mu + torch.exp(0.5*logvar) * torch.randn(x.shape[0], self.Nlatent, device=x.device, generator=_rng)

        # compute negative KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1) 

        return z, KLD
    #}}}
