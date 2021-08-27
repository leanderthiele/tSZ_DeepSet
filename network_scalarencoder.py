import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from global_fields import GlobalFields
from basis import Basis
from default_from_cfg import DefaultFromCfg
import normalization
import cfg


class NetworkScalarEncoder(nn.Module) :
    """
    transforms halo-scale information into a feature vector of scalars
    """

    def __init__(self, Nlatent=DefaultFromCfg('SCALAR_ENCODER_NLATENT'),
                       Nlayers=DefaultFromCfg('SCALAR_ENCODER_NLAYERS'),
                       Nhidden=DefaultFromCfg('SCALAR_ENCODER_NHIDDEN'),
                       basis_passed=DefaultFromCfg('SCALAR_ENCODER_BASIS_PASSED'),
                       globals_passed=DefaultFromCfg('SCALAR_ENCODER_GLOBALS_PASSED'),
                       MLP_kwargs_dict=dict(),
                       **MLP_kwargs) :
        """
        Nlatent ... number of neurons in output
        Nlayers ... number of hidden layers (i.e. #MLPs - 1) [only 0 or 1 really make sense here]
        Nhidden ... number of hidden neurons between MLPs [since only 0 or 1 are useful for Nlayers,
                                                           no complicated dict structure necessary]
        basis_passed   ... whether the basis will be passed
        globals_passed ... whether the global scalars will be passed
        MLP_kwargs_dict ... a dict indexed by str(layer_index) -- does not need to have all keys
                            note that the indices here can be one more than the Nhidden indices
                            NOTE 'first' and 'last' are special keywords that can also be used
        MLP_kwargs      ... default values for the MLP kwargs, can by overriden by specific entries
                            in MLP_kwargs_dict
        """
    #{{{
        if isinstance(Nlatent, DefaultFromCfg) :
            Nlatent = Nlatent()
        if isinstance(Nlayers, DefaultFromCfg) :
            Nlayers = Nlayers()
        if isinstance(Nhidden, DefaultFromCfg) :
            Nhidden = Nhidden()
        if isinstance(basis_passed, DefaultFromCfg) :
            basis_passed = basis_passed()
        if isinstance(globals_passed, DefaultFromCfg) :
            globals_passed = globals_passed()

        super().__init__()

        # figure out number of input features
        k_in = (# radial distances
                1
                # velocity moduli
                + cfg.USE_VELOCITIES
                # global scalars
                + globals_passed * len(GlobalFields)
                # basis projections
                + basis_passed * (1+cfg.USE_VELOCITIES) * len(Basis))

        self.layers = nn.ModuleList(
            [NetworkMLP(k_in if ii==0 else Nhidden,
                        Nlatent if ii==Nlayers else Nhidden,
                        **(MLP_kwargs_dict[str(ii)] if str(ii) in MLP_kwargs_dict \
                           else MLP_kwargs_dict['first'] if 'first' in MLP_kwargs_dict and ii==0 \
                           else MLP_kwargs_dict['last'] if 'last' in MLP_kwargs_dict and ii==Nlayers \
                           else MLP_kwargs)
                       ) for ii in range(Nlayers+1)])

        self.basis_passed = basis_passed
        self.globals_passed = globals_passed
        self.Nlatent = Nlatent
    #}}}


    def create_scalars(self, x, v=None, u=None, basis=None) :
        """
        this function very closely mirrors the one in NetworkLayer, but there are small
        differences so we'll have to live with some code duplication here unfortunately
        if we don't want it to get too confusing
        """
    #{{{
        # descriptive string
        desc = 'input to NetworkScalarEncoder: '

        # get the radial distances
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-5

        scalars = normalization.encoder_x(x_norm)
        desc += '|x| [1]; '

        if basis is not None :
            assert self.basis_passed and len(Basis) != 0
            scalars = torch.cat((scalars,
                                 normalization.unit_contraction(torch.einsum('bid,bnd->bin',
                                                                             x/x_norm,
                                                                             basis))),
                                 dim=-1)
            desc += 'x.basis [%d]; '%len(Basis)
        else :
            assert not self.basis_passed or len(Basis) == 0

        if u is not None :
            assert self.globals_passed and len(GlobalFields) != 0
            scalars = torch.cat((scalars, u.unsqueeze(1).expand(-1, scalars.shape[1], -1)), dim=-1)
            desc += 'u [%d]; '%len(GlobalFields)
        else :
            assert not self.globals_passed or len(GlobalFields) == 0

        if v is not None :
            assert cfg.USE_VELOCITIES
            v_norm = torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-5
            scalars = torch.cat((scalars, normalization.encoder_v(v_norm)), dim=-1)
            desc += '|v| [1]; '
            if basis is not None :
                scalars = torch.cat((scalars,
                                     normalization.unit_contraction(torch.einsum('bid,bnd->bin',
                                                                                 v/v_norm,
                                                                                 basis))),
                                    dim=-1)
                desc += 'v.basis [%d]; '%len(Basis)

        return scalars, desc
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
            return torch.cat([self(xi,
                                   v[ii],
                                   u[ii, ...].unsqueeze(0) if u is not None else u,
                                   basis[ii, ...].unsqueeze(0) if basis is not None else basis)
                              for ii, xi in enumerate(x)])

        scalars, _ = self.create_scalars(x, v, u, basis)

        for ii, l in enumerate(self.layers) :
            
            scalars = l(scalars)

            if ii == 0 :
                # in this case we also need a pooling operation
                scalars = self.__pool(scalars)

        # return shape [batch, N_TNG, N_features]
        return scalars
    #}}}


    def __pool(self, x) :
        """
        input tensor is of shape [batch, N_DM, N_features]
        --> we return [batch, N_features]
        """
    #{{{
        return torch.mean(x, dim=2)
    #}}}
