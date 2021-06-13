import torch
import torch.nn as nn

from network_layer import NetworkLayer
import cfg


class NetworkEncoder(nn.Sequential) :
    """
    transforms the input into a latent space representation
    which we can than evaluate at specific points
    """

    def __init__(self, k_latent,
                       Nlayers=cfg.ENCODER_DEFAULT_NLAYERS,
                       Nhidden=cfg.ENCODER_DEFAULT_NHIDDEN,
                       MLP_kwargs_dict=dict(),
                       **MLP_kwargs) :
        """
        k_latent ... number of neurons in the output
        Nlayers  ... number of hidden layers (i.e. hidden h-vectors)
        Nhidden  ... either an integer, which is the number of h-vectors used in the hidden layers,
                     or a dict indexed by str(layer index) -- does not need to have all keys
                     NOTE 'first' and 'last' are special keywords that can also be used
        MLP_kwargs_dict ... a dict indexed by str(layer_index) -- does not need to have all keys
                            note that the indices here can be one more than the Nhidden indices
                            NOTE 'first' and 'last' are special keywords that can also be used
        MLP_kwargs      ... default values for the MLP kwargs, can by overriden by specific entries
                            in MLP_kwargs_dict
        """
    #{{{
        assert isinstance(Nhidden, int) or isinstance(Nhidden, dict)

        super().__init__(*[NetworkLayer(0 if ii==0 \
                                        else Nhidden if isinstance(Nhidden, int) \
                                        else Nhidden[str(ii-1)] if str(ii-1) in Nhidden \
                                        else cfg.ENCODER_DEFAULT_NHIDDEN,
                                        k_latent if ii==Nlayers \
                                        else Nhidden if isinstance(Nhidden, int) \
                                        else Nhidden[str(ii)] if str(ii) in Nhidden \
                                        else Nhidden['first'] if 'first' in Nhidden and ii==0 \
                                        else Nhidden['last'] if 'last' in Nhidden and ii==Nlayers-1 \
                                        else cfg.ENCODER_DEFAULT_NHIDDEN,
                                        **(MLP_kwargs_dict[str(ii)] if str(ii) in MLP_kwargs_dict \
                                           else MLP_kwargs_dict['first'] if 'first' in MLP_kwargs_dict and ii==0 \
                                           else MLP_kwargs_dict['last'] if 'last' in MLP_kwargs_dict and ii==Nlayers \
                                           else MLP_kwargs)
                                       ) for ii in range(Nlayers+1)])
    #}}}
