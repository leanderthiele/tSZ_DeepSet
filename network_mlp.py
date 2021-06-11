import torch
import torch.nn as nn

import cfg


class _MLPLayer(nn.Sequential) :
    """
    a single layer perceptron, used as building blocks to build the MLPs
    """

    def __init__(self, Nin, Nout, bias=True, layernorm=True) :
    #{{{ 
        super().__init__(nn.LayerNorm(Nin) if layernorm else nn.Identity(),
                         torch.nn.Linear(Nin, Nout, bias=bias),
                         torch.nn.LeakyReLU())
    #}}}


class NetworkMLP(nn.Sequential) :
    """
    a multi-layer perceptron, stacking a number of _MLPLayers
    """

    def __init__(self, Nin, Nout,
                       Nlayers=cfg.MLP_DEFAULT_NLAYERS,
                       Nhidden=cfg.MLP_DEFAULT_NHIDDEN,
                       **layer_kwargs) :
    #{{{
        if 'layernorm' in layer_kwargs :
            default_layernorm = layer_kwargs['layernorm']
            del layer_kwargs['layernorm']
        else :
            default_layernorm = True

        super().__init__(*[_MLPLayer(Nin if ii==0 else Nhidden,
                                     Nout if ii==Nlayers else Nhidden,
                                     # only apply layer normalization to the hidden states
                                     layernorm=False if ii==0 else default_layernorm,
                                     **layer_kwargs)
                           for ii in range(Nlayers+1)])
    #}}}
