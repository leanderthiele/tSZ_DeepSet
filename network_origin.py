from network_encoder import NetworkEncoder
from default_from_cfg import DefaultFromCfg
import cfg


class NetworkOrigin(NetworkEncoder) :
    """
    predicts a single 3-vector (in this case the origin) from the DM particle positions
    """

    def __init__(self, Nlayers=DefaultFromCfg('ORIGIN_DEFAULT_NLAYERS'),
                       Nhidden=DefaultFromCfg('ORIGIN_DEFAULT_NHIDDEN'),
                       basis_max_layer=DefaultFromCfg('ORIGIN_DEFAULT_BASIS_MAXLAYER'),
                       globals_max_layer=DefaultFromCfg('ORIGIN_DEFAULT_GLOBALS_MAXLAYER'),
                       **MLP_kwargs) :
    #{{{
        if isinstance(Nlayers, DefaultFromCfg) :
            Nlayers = Nlayers()
        if isinstance(Nhidden, DefaultFromCfg) :
            Nhidden = Nhidden()
        if isinstance(basis_max_layer, DefaultFromCfg) :
            basis_max_layer = basis_max_layer()
        if isinstance(globals_max_layer, DefaultFromCfg) :
            globals_max_layer = globals_max_layer()

        super().__init__(Nlatent=2, # predict two vectors
                         # do not have an activation function before the final output
                         # since we generally want to map to the entire real line
                         # initialize last bias to zero because on average we don't expect a shift
                         Nlayers=Nlayers,
                         Nhidden=Nhidden,
                         basis_max_layer=basis_max_layer,
                         globals_max_layer=globals_max_layer,
                         MLP_kwargs_dict=dict(last=dict(layer_kwargs_dict=\
                            dict(last={'activation' : False,
                                       'dropout': None,
                                       'bias_init': 'zeros_(%s)'}))),
                         **MLP_kwargs
                        )
    #}}}


    def to_device(self) :
    #{{{
        if cfg.device_idx is not None :
            return self.to(cfg.device_idx)
        else :
            return self
    #}}}
