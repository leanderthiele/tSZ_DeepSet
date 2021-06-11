import torch
import torch.nn as nn

from network_mlp import NetworkMLP
import cfg


class NetworkDecoder(nn.Module) :
    """
    takes the latent space vectors and the positions where we want to evaluate
    and gives us scalar outputs at the requested locations
    (if requested, k_out can be chosen as non-unity to have multiple values at those
     locations, but they will no longer be vector valued!)
    """

    def __init__(self, k_latent, k_out=1, **MLP_kwargs) :
        """
        k_latent   ... the number of latent space vectors
        k_out      ... the number of features to predict at the locations
        MLP_kwargs ... to specify the multi-layer perceptron used here
        """
    #{{{
        super().__init__()

        self.mlp = NetworkMLP(k_latent+cfg.NGLOBALS, k_out, **MLP_kwargs)
    #}}}


    def forward(self, h, x, u=None) :
        """
        h ... the latent vectors, of shape [batch, latent feature, 3]
        x ... the positions where to evaluate, of shape [batch, Nvects, 3]
        u ... the global vector, of shape [batch, Nglobals]
        """
    #{{{
        # compute the projections of shape [batch, Nvects, latent feature]
        projections = torch.einsum('bvd,bld->bvl', x, h)

        # concatenate with the global vector if requested
        if u is not None :
            assert len(u) == cfg.NGLOBALS
            projections = torch.cat((u.unsqueeze(1).repeat(1,projections.shape[1],1), projections), dim=-1)
        else :
            assert not cfg.NGLOBALS

        # pass through the MLP, transform scalars -> scalars
        return self.mlp(projections)
    #}}}
