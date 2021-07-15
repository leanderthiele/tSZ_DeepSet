from network_encoder import NetworkEncoder
from default_from_cfg import DefaultFromCfg, SetDefaults
import cfg


class NetworkOrigin(NetworkEncoder) :
    """
    predicts a single 3-vector (in this case the origin) from the DM particle positions
    """

    def __init__(self, Nlayers=DefaultFromCfg('ORIGIN_DEFAULT_NLAYERS')) :
    #{{{
        exec(SetDefaults(locals()))

        super().__init__(1, # predict exactly one vector
                         # do not have an activation function before the final output
                         # since we generally want to map to the entire real line
                         # initialize last bias to zero because on average we don't expect a shift
                         Nlayers=Nlayers,
                         MLP_kwargs_dict=dict(last=dict(layer_kwargs_dict=dict(last={'activation' : False,
                                                                                     'dropout': None,
                                                                                     'bias_init': 'zeros_(%s)'})))
                        )
    #}}}


    def to_device(self) :
    #{{{
        if cfg.device_idx is not None :
            return self.to(cfg.device_idx)
        else :
            return self
    #}}}
