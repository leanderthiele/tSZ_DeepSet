import torch
import torch.nn as nn

import cfg


class _MLPLayer(nn.Sequential) :
    """
    a single layer perceptron, used as building blocks to build the MLPs
    """

    def __init__(self, Nin, Nout, bias=True) :
    #{{{ 
        super().__init__(torch.nn.Linear(Nin, Nout, bias=bias),
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
        super().__init__(*[_MLPLayer(Nin if ii==0 else Nhidden,
                                     Nout if ii==Nlayers else Nhidden,
                                     **layer_kwargs)
                           for ii in range(Nlayers+1)])
    #}}}
