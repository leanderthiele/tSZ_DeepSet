import math

import torch
import torch.nn as nn

import cfg


class _MLPLayer(nn.Sequential) :
    """
    a single layer perceptron, used as building blocks to build the MLPs
    """

    def __init__(self, Nin, Nout, input_is_hidden,
                       bias=True, layernorm=cfg.LAYERNORM,
                       activation=True, dropout=cfg.DROPOUT) :
        """
        set dropout to None if not desired
        """
    #{{{ 
        # NOTE apparently the ordering here is not that well explored, but I found at least
        #      one source that says a Google network has dropout after layer normalization

        super().__init__(nn.LayerNorm(Nin) if layernorm and input_is_hidden else nn.Identity(),
                         nn.Dropout(p=dropout) if input_is_hidden and dropout is not None \
                         else nn.Dropout(p=cfg.VISIBLE_DROPOUT) if dropout is not None and cfg.VISIBLE_DROPOUT is not None \
                         else nn.Identity(),
                         nn.Linear(Nin, Nout, bias=bias),
                         nn.LeakyReLU() if activation else nn.Identity())

        # NOTE be careful about the indexing here if the sequential order is changed
        nn.init.kaiming_uniform_(self[2].weight,
                                 a=self[3].negative_slope if activation else math.sqrt(5)
                                )
        if bias :
            nn.init.ones_(self[2].bias)
    #}}}


class NetworkMLP(nn.Sequential) :
    """
    a multi-layer perceptron, stacking a number of _MLPLayers
    """

    def __init__(self, Nin, Nout,
                       MLP_Nlayers=cfg.MLP_DEFAULT_NLAYERS,
                       MLP_Nhidden=cfg.MLP_DEFAULT_NHIDDEN,
                       layer_kwargs_dict=dict(),
                       **layer_kwargs) :
        """
        Nin  ... number of features coming in
        Nout ... number of features going out
        MLP_Nlayers ... number of hidden layers
        MLP_Nhidden ... number of hidden neurons, either an integer which is used universally
                        or a dict indexed by str(layer_index) -- does not need to have all keys
                        NOTE 'first' and 'last' are special keywords that can also be used
        layer_kwargs_dict ... a dict indexed by str(layer_index) -- does not need to have all keys
                              note that the indices here can be one more than the Nhidden indices
                              NOTE 'first' and 'last' are special keywords that can also be used
        layer_kwargs      ... default values for layer kwargs, can be overriden by specific entries
                              in layer_kwargs_dict
        """
    #{{{
        super().__init__(*[_MLPLayer(Nin if ii==0 else MLP_Nhidden,
                                     Nout if ii==MLP_Nlayers else MLP_Nhidden,
                                     ii != 0, # = input is hidden
                                     **(layer_kwargs_dict[str(ii)] if str(ii) in layer_kwargs_dict \
                                        else layer_kwargs_dict['first'] if 'first' in layer_kwargs_dict and ii==0 \
                                        else layer_kwargs_dict['last'] if 'last' in layer_kwargs_dict and ii==MLP_Nlayers \
                                        else layer_kwargs))
                           for ii in range(MLP_Nlayers+1)])
    #}}}
