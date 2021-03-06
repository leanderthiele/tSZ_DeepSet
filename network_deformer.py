
import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from basis import Basis
from global_fields import GlobalFields
import normalization
from default_from_cfg import DefaultFromCfg
import cfg


class NetworkDeformer(nn.Module) :
    """
    takes the TNG particle positiions and returns altered radii depending
    on some global properties of the halo
    """

    def __init__(self, globals_passed=DefaultFromCfg('DEFORMER_GLOBALS_PASSED'),
                       **MLP_kwargs) :
    #{{{ 
        if isinstance(globals_passed, DefaultFromCfg) :
            globals_passed = globals_passed()
    
        super().__init__()

        self.mlp = NetworkMLP(1 # radial coordinate
                              + globals_passed * len(GlobalFields)
                              + len(Basis),
                              1, # output only the changed radial magnitude
                              **MLP_kwargs)

        self.globals_passed = globals_passed
    #}}}


    def create_scalars(self, x, r, u=None, basis=None) :
    #{{{ 
        desc = 'input to NetworkDeformer: '

        scalars = normalization.TNG_radii(r)
        desc += 'r [1]; '

        # concatenate with the basis projections if required
        if basis is not None :
            scalars = torch.cat((scalars,
                                 normalization.unit_contraction(torch.einsum('bid,bnd->bin',
                                                                             x/(r + 1e-5),
                                                                             basis))),
                                dim=-1)
            desc += 'x.basis [%d]; '%len(Basis)

        # concatenate with the global features if required
        if u is not None :
            assert self.globals_passed and len(GlobalFields) > 0
            scalars = torch.cat((scalars, u.unsqueeze(1).expand(-1, scalars.shape[1], -1)), dim=-1)
            desc += 'u [%d]; '%len(GlobalFields)
        else :
            assert not self.globals_passed or len(GlobalFields) == 0

        return scalars, desc
    #}}}


    def forward(self, x, r, u=None, basis=None) :
        """
        x ... the TNG particle coordinates, either of shape [batch, Npart, 3]
              or a list of shapes [1, Nparti, 3]
        r ... the TNG particle radial coordinates, either of shape [batch, Npart, 1]
              or a list of shapes [1, Nparti, 1]
        u ... the global features, of shape [batch, Nglobals]
        basis ... the basis vectors to use, of shape [batch, Nbasis, 3]
                  (note that it doesn't make much sense not to pass a basis since then
                   we have isotropy and x is not used)
        """
    #{{{
        if isinstance(x, list) :
            return [self(xi, r[ii],
                         u[ii, ...].unsqueeze(0) if u is not None else u,
                         basis[ii, ...].unsqueeze(0) if basis is not None else basis)
                    for ii, xi in enumerate(x)]

        scalars, _ = self.create_scalars(x, r, u, basis)

        # pass through the MLP
        scalars = self.mlp(scalars)

        # we take the exponential to ensure that the output is strictly positive,
        # otherwise we'll get divergences when evaluating the Battaglia model
        # NOTE also that the exponential is unity at zero, which is a useful feature
        return torch.exp(scalars) * r
    #}}}
