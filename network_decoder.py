import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from global_fields import GlobalFields
from basis import Basis
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

        self.mlp = NetworkMLP(k_latent+len(GlobalFields)+len(Basis), k_out, **MLP_kwargs)
    #}}}


    def forward(self, h, x, u=None, basis=None) :
        """
        h ... the latent vectors, of shape [batch, latent feature, 3]
        x ... the positions where to evaluate, of shape [batch, Nvects, 3]
              or a list of length batch and shapes [1, Nvectsi, 3]
        u ... the global vector, of shape [batch, Nglobals]
        basis ... the basis vectors to use -- either None if no basis is provided
                  or of shape [batch, Nbasis, 3]
        """
    #{{{
        if isinstance(x, list) :
            return [self(h[ii, ...].unsqueeze(0),
                         xi,
                         u[ii, ...].unsqueeze(0) if u is not None else u,
                         basis[ii, ...].unsqueeze(0) if basis is not None else basis)
                    for ii, xi in enumerate(x)]

        # compute the projections of shape [batch, Nvects, latent feature]
        projections = torch.einsum('bvd,bld->bvl', x, h)

        # concatenate with the basis projections if needed
        if basis is not None :
            basis_projections = torch.einsum('bid,bnd->bin', x, basis)
            projections = torch.cat((projections, basis_projections), dim=-1)
        else :
            assert len(Basis) == 0

        # concatenate with the global vector if requested
        if u is not None :
            projections = torch.cat((u.unsqueeze(1).repeat(1,projections.shape[1],1), projections), dim=-1)
        else :
            assert len(GlobalFields) == 0

        # pass through the MLP, transform scalars -> scalars
        return self.mlp(projections)
    #}}}
