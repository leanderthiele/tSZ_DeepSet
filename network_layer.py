import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from global_fields import GlobalFields
from basis import Basis
import cfg


class NetworkLayer(nn.Module) :
    """
    one block of the network, transforms a set of vectors into another set of vectors
    """

    def __init__(self, k_in, k_out, velocities_passed=True, basis_passed=True, globals_passed=True, **MLP_kwargs) :
        """
        Nk_in  ... number of feature vectors coming in
                   (set to non-positive for the initial layer that takes the positions)
        Nk_out ... number of neurons going out
        velocities_passed ... whether the particle velocities are going to be passed
        basis_passed ... whether the basis will be passed
        globals_passed ... whether the globals will be passed
        MLP_kwargs ... kwargs that will be passed forward to the NetworkMLP constructor
        """
    #{{{
        super().__init__()

        # whether the forward method takes some latent vectors or actual particle positions
        # (in the latter case, we do not compute all the mutual dot products)
        self.x_is_latent = k_in > 0

        if self.x_is_latent :
            assert not velocities_passed

        self.mlp = NetworkMLP(1 + velocities_passed
                                + globals_passed * len(GlobalFields)
                                + (1+velocities_passed)*(basis_passed * len(Basis))
                                + self.x_is_latent * k_in,
                              k_out, **MLP_kwargs)

        self.velocities_passed = velocities_passed
        self.globals_passed = globals_passed
        self.basis_passed = basis_passed
    #}}}


    def forward(self, x, v=None, u=None, basis=None) :
        """
        x     ... the input positions tensor, of shape [batch, Nvecs, 3]
                  or a list of length batch with shapes [1, Nvecs[ii], 3]
        v     ... the input velocities tensor, of shape [batch, Nvecs, 3]
                  or a list of length batch with shapes [1, Nvecs[ii], 3]
        u     ... the global tensor -- assumed to be a vector, i.e. of shape [batch, Nglobals]
        basis ... the basis vectors to use -- either None if no basis is provided
                  or of shape [batch, Nbasis, 3]
        """
    #{{{
        if isinstance(x, list) :
            assert not self.x_is_latent # this can only happen for the variable size initial inputs
            return torch.cat([self(xi,
                                   v[ii],
                                   u[ii, ...].unsqueeze(0) if u is not None else u,
                                   basis[ii, ...].unsqueeze(0) if basis is not None else basis)
                              for ii, xi in enumerate(x)])

        # we know that the last dimension is the real space one
        norms = torch.linalg.norm(x, dim=-1, keepdim=True)

        # this tensor will collect all scalar quantities
        scalars = norms.clone()

        # concatenate with the basis projections if required
        if basis is not None :
            assert self.basis_passed and len(Basis) != 0
            basis_projections = torch.einsum('bid,bnd->bin', x, basis) / norms
            scalars = torch.cat((scalars, basis_projections), dim=-1)
        else :
            assert not self.basis_passed or len(Basis) == 0

        if self.x_is_latent :
            # compute the mutual dot products
            dots = torch.einsum('bid,bjd->bij', x, x)
            scalars = torch.cat((scalars, dots), dim=-1)
        else :
            # we are in the very first layer and need to normalize the vector
            x = x / norms

        # concatenate with the global vector if requested
        if u is not None :
            assert self.globals_passed and len(GlobalFields) != 0
            scalars = torch.cat((u.unsqueeze(1).repeat(1, scalars.shape[1], 1), scalars), dim=-1)
        else :
            assert not self.globals_passed or len(GlobalFields) == 0

        # if passed, concatenate with velocity magnitudes and, if a basis is given,
        # with the projections on the basis vectors
        if v is not None :
            assert self.velocities_passed and cfg.USE_VELOCITIES
            norms = torch.linalg.norm(v, dim=-1, keepdim=True)
            scalars = torch.cat((scalars, norms), dim=-1)
            if basis is not None :
                basis_projections = torch.einsum('bid,bnd->bin', v, basis) / norms
                scalars = torch.cat((scalars, basis_projections), dim=-1)

        # pass through the MLP, transform scalars -> scalars
        fk = self.mlp(scalars)

        # evaluate in the vector space
        vecs = torch.einsum('bio,bid->biod', fk, x)

        # return the pooled version
        return self.__pool(vecs)
    #}}}


    def __pool(self, x) :
        """
        input tensor is of shape [batch, input Nvecs, output Nvecs, 3]
        """
    #{{{
        return torch.mean(x, dim=1)
    #}}}
