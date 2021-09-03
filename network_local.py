import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from basis import Basis
import normalization
from default_from_cfg import DefaultFromCfg
from merge_dicts import MergeDicts
import cfg


class NetworkLocal(nn.Module) :
    """
    transforms the local environment into a vector of scalars
    """


    def __init__(self, Nlatent=DefaultFromCfg('LOCAL_NLATENT'),
                       Nlayers=DefaultFromCfg('LOCAL_NLAYERS'),
                       Nhidden=DefaultFromCfg('LOCAL_NHIDDEN'),
                       pass_N=DefaultFromCfg('LOCAL_PASS_N'),
                       concat_with_N=DefaultFromCfg('LOCAL_CONCAT_WITH_N'),
                       MLP_kwargs_dict=dict(),
                       **MLP_kwargs) :
        """
        Nlatent  ... number of neurons (scalar features) in the output
        Nlayers  ... number of hidden layers (each layer is an MLP)
        Nhidden  ... number of neurons in hidden layers (either int or dict)
        pass_N   ... whether the local number of particles is used in the initial DeepSet step
                     or only concatenated with the scalars after the pooling
        concat_with_N ... whether the local number of particles is concatenated with the scalars
                          produced after pooling
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
        if isinstance(pass_N, DefaultFromCfg) :
            pass_N = pass_N()
        if isinstance(concat_with_N, DefaultFromCfg) :
            concat_with_N = concat_with_N()

        assert isinstance(Nhidden, (int, dict))
        assert pass_N or concat_with_N

        super().__init__()

        # figure out the number of input features
        k_in = (# projections of DM positions and velocities, TNG position, bulk velocity
                len(Basis) * (1+cfg.LOCAL_PASS_RELATIVE_TO_HALO) * (1+cfg.USE_VELOCITIES)
                # moduli of DM positions and velocities, TNG position, bulk velocity
                + (1+cfg.LOCAL_PASS_RELATIVE_TO_HALO) * (1+cfg.USE_VELOCITIES)
                # x0.v0 and x.v
                + (1+cfg.LOCAL_PASS_RELATIVE_TO_HALO) * cfg.USE_VELOCITIES
                # number of particles in the sphere
                + pass_N)

        self.layers = nn.ModuleList(
            [NetworkMLP(# number of input features
                        k_in if ii==0 \
                        else Nhidden if isinstance(Nhidden, int) \
                        else Nhidden[str(ii-1)] if str(ii-1) in Nhidden \
                        else Nhidden['first'] if 'first' in Nhidden and ii==1 \
                        else cfg.LOCAL_NHIDDEN,
                        # number of output features
                        # for the first layer, we need to make space for N if concat_with_N
                        (Nlatent if ii==Nlayers \
                         else Nhidden if isinstance(Nhidden, int) \
                         else Nhidden[str(ii)] if str(ii) in Nhidden \
                         else Nhidden['first'] if 'first' in Nhidden and ii==0 \
                         else Nhidden['last'] if 'last' in Nhidden and ii==Nlayers-1 \
                         else cfg.LOCAL_NHIDDEN) - (1 if ii==0 and concat_with_N else 0),
                        # other arguments
                        **(MergeDicts(MLP_kwargs_dict[str(ii)], MLP_kwargs) if str(ii) in MLP_kwargs_dict \
                           else MergeDicts(MLP_kwargs_dict['first'], MLP_kwargs) if 'first' in MLP_kwargs_dict and ii==0 \
                           else MergeDicts(MLP_kwargs_dict['last'], MLP_kwargs) if 'last' in MLP_kwargs_dict and ii==Nlayers \
                           else MLP_kwargs)
                       ) for ii in range(Nlayers+1)])

        self.pass_N = pass_N
        self.concat_with_N = concat_with_N
    #}}}


    def create_scalars(self, x0, x, N, basis, v) :
    #{{{
        # descriptive string identifying the scalars
        desc = 'input to NetworkLocal: '

        # get some shape information
        N_DM  = x.shape[2]

        if cfg.USE_VELOCITIES :
            # compute bulk motion and subtract from the velocities
            v0 = torch.mean(v, dim=2) # [batch, N_TNG, 3]
            v -= v0.unsqueeze(2)

        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-5

        # the local coordinates are normalized to unit cfg.R_LOCAL
        assert x_norm.max().item() < 1.05, x_norm.max().item()

        scalars = normalization.local_x(x_norm)
        desc += '|x| [1]; '

        scalars = torch.cat((scalars,
                             normalization.unit_contraction(torch.einsum('bijd,bkd->bijk',
                                                                         x/x_norm,
                                                                         basis))),
                            dim=-1)
        desc += 'x.basis [%d]; '%len(Basis)

        if self.pass_N :
            # measure of number of DM particles in vicinity, has shape [batch, N_TNG, N_DM, 1]
            N_expanded = normalization.local_N(N).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N_DM, -1)
            scalars = torch.cat((scalars, N_expanded), dim=-1)
            desc += 'N [1]; '

        # one of the TNG particles sits at r = 0, prevent explosions
        x0_norm = torch.linalg.norm(x0, dim=-1, keepdim=True) + 1e-5

        if cfg.LOCAL_PASS_RELATIVE_TO_HALO :
            scalars = torch.cat((scalars,
                                 normalization.TNG_radii(x0_norm).unsqueeze(-1).expand(-1, -1, N_DM, -1)),
                                dim = -1)
            desc += '|x0| [1]; '

        if cfg.LOCAL_PASS_RELATIVE_TO_HALO :
            scalars = torch.cat((scalars,
                                 normalization.unit_contraction(torch.einsum('bid,bjd->bij',
                                                                x0/x0_norm,
                                                                basis)).unsqueeze(2).expand(-1, -1, N_DM, -1)),
                                dim=-1)
            desc += 'x0.basis [%d]; '%len(Basis)

        if cfg.USE_VELOCITIES and cfg.LOCAL_PASS_RELATIVE_TO_HALO :
            v0_norm = torch.linalg.norm(v0, dim=-1, keepdim=True) + 1e-5
            scalars = torch.cat((scalars,
                                 normalization.local_v0(v0_norm).unsqueeze(-1).expand(-1, -1, N_DM, -1)),
                                dim = -1)
            desc += '|v0| [1]; '

            scalars = torch.cat((scalars,
                                 normalization.unit_contraction(torch.einsum('bid,bjd->bij',
                                                                v0/v0_norm,
                                                                basis)).unsqueeze(2).expand(-1, -1, N_DM, -1)),
                                dim=-1)
            desc += 'v0.basis [%d]; '%len(Basis)

            scalars = torch.cat((scalars,
                                 normalization.unit_contraction(torch.einsum('bid,bid->bi',
                                                                x0/x0_norm,
                                                                v0/v0_norm)).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N_DM, -1)),
                                dim=-1)
            desc += 'x0.v0 [1]; '


        if cfg.USE_VELOCITIES :
            v_norm = torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-5
            scalars = torch.cat((scalars, normalization.local_v(v_norm)), dim=-1)
            desc += '|v| [1]; '

            scalars = torch.cat((scalars,
                                 normalization.unit_contraction(torch.einsum('bijd,bkd->bijk',
                                                                             v/v_norm,
                                                                             basis))),
                                dim=-1)
            desc += 'v.basis [%d]; '%len(Basis)

            scalars = torch.cat((scalars,
                                 normalization.unit_contraction(torch.einsum('bijd,bijd->bij',
                                                                             x/x_norm,
                                                                             v/v_norm)).unsqueeze(-1)),
                                dim=-1)
            desc += 'x.v [1]; '

        return scalars, desc
    #}}}


    def forward(self, x0, x, N, basis, v, P200c) :
        """
        x0    ... the TNG positions around which we are evaluating,
                  of shape [batch, N_TNG, 3] or a list of length batch with shapes [1, N_TNG[ii], 3]
        x     ... the DM positions (in the local frame, i.e. relative to x0),
                  of shape [batch, N_TNG, N_DM, 3] or a list of length batch with shapes [1, N_TNG[ii], N_DM, 3]
        N     ... number of DM particles in vicinity of TNG position [not identical to len(x)!],
                  of shape [batch, N_TNG] or a list of length batch with shapes [1, N_TNG[ii]]
        basis ... the usual global vectors,
                  of shape [batch, Nbasis, 3]
        v     ... the DM velocities (in the halo frame),
                  of shape [batch, N_TNG, N_DM, 3] or a list of length batch with shapes [1, N_TNG[ii], N_DM, 3]
        P200c ... used to divide the network output in the end (if cfg.SCALE_PTH), this way the network doesn't
                  need to learn anything about the scaling of the pressure. shape [batch]

        NOTE we currently limit this function to the case that for each halo the number of DM particles N_DM per
             TNG particle is identical
        """
    #{{{
        if isinstance(x0, list) :
            return [self(x0i,
                         x[ii],
                         N[ii],
                         basis[ii].unsqueeze(0),
                         v[ii] if v is not None else v,
                         torch.tensor([P200c[ii],], requires_grad=False))
                    for ii, x0i in enumerate(x0)]

        assert (v is None and not cfg.USE_VELOCITIES) \
               or (v is not None and cfg.USE_VELOCITIES)

        scalars, _ = self.create_scalars(x0, x, N, basis, v)

        for ii, l in enumerate(self.layers) :
            
            scalars = l(scalars)

            if ii == 0 :
                # in this case we also need a pooling operation
                scalars = self.__pool(scalars)

                if self.concat_with_N :
                    # and we need to concatenate with the local DM density measure
                    scalars = torch.cat((scalars, normalization.local_N(N).unsqueeze(-1)), dim=-1)

        # return shape [batch, N_TNG, N_features]
        return (1 if not cfg.SCALE_PTH else (1/P200c)[:, None, None]) \
               * scalars
    #}}}


    def __pool(self, x) :
        """
        input tensor is of shape [batch, N_TNG, N_DM, N_features]
        --> we return [batch, N_TNG, N_features]
        """
    #{{{
        return torch.mean(x, dim=2)
    #}}}
