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

    def __init__(self, k_latent, k_out=1,
                 r_passed=cfg.DECODER_DEFAULT_R_PASSED,
                 basis_passed=cfg.DECODER_DEFAULT_BASIS_PASSED,
                 globals_passed=cfg.DECODER_DEFAULT_GLOBALS_PASSED,
                 **MLP_kwargs) :
        """
        k_latent   ... the number of latent space vectors (can be zero, then h should be None in forward)
        k_out      ... the number of features to predict at the locations
        r_passed   ... whether the TNG radial coordinates will be passed
        basis_passed   ... whether the basis vectors will be passed
        globals_passed ... whether the globals will be passed
        MLP_kwargs ... to specify the multi-layer perceptron used here
        """
    #{{{
        super().__init__()

        self.mlp = NetworkMLP(k_latent + r_passed
                              + globals_passed * len(GlobalFields)
                              + basis_passed * len(Basis),
                              k_out, **MLP_kwargs)

        self.r_passed = r_passed
        self.globals_passed = globals_passed
        self.basis_passed = basis_passed
    #}}}


    def forward(self, x, h=None, r=None, u=None, basis=None) :
        """
        x ... the positions where to evaluate, of shape [batch, Nvects, 3]
              or a list of length batch and shapes [1, Nvectsi, 3]
        h ... the latent vectors, of shape [batch, latent feature, 3]
        r ... the radial positions where to evaluate, of shape [batch, Nvects, 1]
              or a list of length batch shapes [1, Nvectsi, 1]
        u ... the global vector, of shape [batch, Nglobals]
        basis ... the basis vectors to use -- either None if no basis is provided
                  or of shape [batch, Nbasis, 3]
        """
    #{{{
        if isinstance(x, list) :
            return [self(h[ii, ...].unsqueeze(0),
                         xi,
                         r=r[ii] if r is not None else r,
                         u=u[ii, ...].unsqueeze(0) if u is not None else u,
                         basis=basis[ii, ...].unsqueeze(0) if basis is not None else basis)
                    for ii, xi in enumerate(x)]

        scalars = None

        if h is not None :
            # compute the projections of shape [batch, Nvects, latent feature]
            scalars = torch.einsum('bvd,bld->bvl', x, h)

        # concatenate with the radial distances if needed
        if r is not None :
            assert self.r_passed
            scalars = torch.cat((scalars, r), dim=-1) if scalars is not None \
                      else r.clone()
        else :
            assert not self.r_passed

        # concatenate with the basis projections if needed
        if basis is not None :
            assert self.basis_passed and len(Basis) != 0
            basis_projections = torch.einsum('bid,bnd->bin', x, basis)
            scalars = torch.cat((scalars, basis_projections), dim=-1) if scalars is not None \
                          else basis_projections
        else :
            assert not self.basis_passed or len(Basis) == 0

        # concatenate with the global vector if requested
        if u is not None :
            assert self.globals_passed and len(GlobalFields) != 0
            u_repeated = u.unsqueeze(1).repeat(1, x.shape[1], 1)
            scalars = torch.cat((u_repeated, scalars), dim=-1) \
                      if scalars is not None \
                      else u_repeated
        else :
            assert not self.globals_passed or len(GlobalFields) == 0

        # pass through the MLP, transform scalars -> scalars
        return self.mlp(scalars)
    #}}}
