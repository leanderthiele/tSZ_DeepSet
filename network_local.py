import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from basis import Basis
from global_fields import GlobalFields
from default_from_cfg import DefaultFromCfg
import cfg


class NetworkLocal(nn.Module) :
    """
    transforms the local environment into a vector of scalars
    """


    def __init__(self, k_latent,
                       Nlayers=DefaultFromCfg('LOCAL_DEFAULT_NLAYERS'),
                       Nhidden=DefaultFromCfg('LOCAL_DEFAULT_NHIDDEN'),
                       MLP_kwargs_dict=dict(),
                       **MLP_kwargs) :
        """
        k_latent ... number of neurons (scalar features) in the output
        Nlayers  ... number of hidden layers (each layer is an MLP)
        Nhidden  ... number of neurons in hidden layers (either int or dict)
        MLP_kwargs_dict ... a dict indexed by str(layer_index) -- does not need to have all keys
                            note that the indices here can be one more than the Nhidden indices
                            NOTE 'first' and 'last' are special keywords that can also be used
        MLP_kwargs      ... default values for the MLP kwargs, can by overriden by specific entries
                            in MLP_kwargs_dict
        """
    #{{{
        if isinstance(Nlayers, DefaultFromCfg) :
            Nlayers = Nlayers()
        if isinstance(Nhidden, DefaultFromCfg) :
            Nhidden = Nhidden()

        assert isinstance(Nhidden, int) or isinstance(Nhidden, dict)

        super().__init__()

        # figure out the number of input features
        k_in = (len(Basis) * 2 * (1+cfg.USE_VELOCITIES) # projections of DM positions and velocities, TNG position, bulk velocity
                + 2 * (1+cfg.USE_VELOCITIES) # moduli of DM positions and velocities, TNG position, bulk velocity
                + 1) # number of particles in the sphere

        self.layers = nn.ModuleList(
            [NetworkMLP(k_in if ii==0 \
                        else Nhidden if isinstance(Nhidden, int) \
                        else Nhidden[str(ii-1)] if str(ii-1) in Nhidden \
                        else Nhidden['first'] if 'first' in Nhidden and ii==1 \
                        else cfg.LOCAL_DEFAULT_NHIDDEN,
                        k_latent if ii==Nlayers \
                        else Nhidden if isinstance(Nhidden, int) \
                        else Nhidden[str(ii)] if str(ii) in Nhidden \
                        else Nhidden['first'] if 'first' in Nhidden and ii==0 \
                        else Nhidden['last'] if 'last' in Nhidden and ii==Nlayers-1 \
                        else cfg.LOCAL_DEFAULT_NHIDDEN,
                        input_is_hidden=ii != 0,
                        **(MLP_kwargs_dict[str(ii)] if str(ii) in MLP_kwargs_dict \
                           else MLP_kwargs_dict['first'] if 'first' in MLP_kwargs_dict and ii==0 \
                           else MLP_kwargs_dict['last'] if 'last' in MLP_kwargs_dict and ii==Nlayers \
                           else MLP_kwargs)
                       ) for ii in range(Nlayers+1)])
    #}}}


    def forward(self, x0, x, N, basis, v=None) :
        """
        x0    ... the TNG positions around which we are evaluating,
                  of shape [batch, N_TNG, 3] or a list of length batch with shapes [1, N_TNG[ii], 3]
        x     ... the DM positions (in the halo frame),
                  of shape [batch, N_TNG, N_DM, 3] or a list of length batch with shapes [1, N_TNG[ii], N_DM, 3]
        N     ,.. number of DM particles in vicinity of TNG position [not identical to len(x)!],
                  of shape [batch, N_TNG] or a list of length batch with shapes [1, N_TNG[ii]]
        basis ... the usual global vectors,
                  of shape [batch, Nbasis, 3]
        v     ... the DM velocities (in the halo frame),
                  of shape [batch, N_TNG, N_DM, 3] or a list of length batch with shapes [1, N_TNG[ii], N_DM, 3]

        NOTE we currently limit this function to the case that for each halo the number of DM particles N_DM per
             TNG particle is identical
        """
    #{{{
        # TODO
    #}}}
