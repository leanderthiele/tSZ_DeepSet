
from network_encoder import NetworkEncoder
import cfg


class NetworkOrigin(NetworkEncoder) :
    """
    predicts a single 3-vector (in this case the origin) from the DM particle positions
    """

    def __init__(self) :
    #{{{
        super().__init__(1, # predict exactly one vector
                         Nlayers=0,
                         # do not have an activation function before the final output
                         # since we generally want to map to the entire real line
                         MLP_kwargs_dict=dict(last=dict(layer_kwargs_dict=dict(last={'activation' : False,
                                                                                     'dropout': None})))
                        )
    #}}}


    def to_device(self) :
    #{{{
        if cfg.DEVICE_IDX is not None :
            return self.to(cfg.DEVICE_IDX)
        else :
            return self
    #}}}
