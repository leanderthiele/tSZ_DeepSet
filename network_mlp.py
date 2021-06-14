import math

import torch
import torch.nn as nn

import cfg


class _MLPLayer(nn.Sequential) :
    """
    a single layer perceptron, used as building blocks to build the MLPs
    """

    def __init__(self, Nin, Nout, bias=True, layernorm=cfg.LAYERNORM, activation=True) :
    #{{{ 
        super().__init__(nn.LayerNorm(Nin) if layernorm else nn.Identity(),
                         torch.nn.Linear(Nin, Nout, bias=bias),
                         torch.nn.LeakyReLU() if activation else nn.Identity())

        nn.init.kaiming_uniform_(self[1].weight,
                                 a=self[2].negative_slope if activation else math.sqrt(5)
                                )
        if bias :
            nn.init.ones_(self[1].bias)
    #}}}


class NetworkMLP(nn.Sequential) :
    """
    a multi-layer perceptron, stacking a number of _MLPLayers
    """

    def __init__(self, Nin, Nout,
                       Nlayers=cfg.MLP_DEFAULT_NLAYERS,
                       Nhidden=cfg.MLP_DEFAULT_NHIDDEN,
                       layer_kwargs_dict=dict(),
                       **layer_kwargs) :
        """
        Nin  ... number of features coming in
        Nout ... number of features going out
        Nlayers ... number of hidden layers
        Nhidden ... number of hidden neurons, either an integer which is used universally
                    or a dict indexed by str(layer_index) -- does not need to have all keys
                    NOTE 'first' and 'last' are special keywords that can also be used
        layer_kwargs_dict ... a dict indexed by str(layer_index) -- does not need to have all keys
                              note that the indices here can be one more than the Nhidden indices
                              NOTE 'first' and 'last' are special keywords that can also be used
        layer_kwargs      ... default values for layer kwargs, can be overriden by specific entries
                              in layer_kwargs_dict
        """
    #{{{
        if 'layernorm' in layer_kwargs :
            default_layernorm = layer_kwargs['layernorm']
            layer_kwargs.pop('layernorm')
        else :
            default_layernorm = cfg.LAYERNORM

        NetworkMLP.__remove_layer_norm(layer_kwargs_dict, 0)
        NetworkMLP.__remove_layer_norm(layer_kwargs_dict, Nlayers)
        NetworkMLP.__remove_layer_norm(layer_kwargs_dict, 'first')
        NetworkMLP.__remove_layer_norm(layer_kwargs_dict, 'last')

        super().__init__(*[_MLPLayer(Nin if ii==0 else Nhidden,
                                     Nout if ii==Nlayers else Nhidden,
                                     # only apply layer normalization to the hidden states
                                     layernorm=False if ii==0 or ii==Nlayers \
                                               else layer_kwargs_dict[str(ii)]['layernorm'] if str(ii) in layer_kwargs_dict \
                                               else default_layernorm,
                                     **(layer_kwargs_dict[str(ii)] if str(ii) in layer_kwargs_dict \
                                        else layer_kwargs_dict['first'] if 'first' in layer_kwargs_dict and ii==0 \
                                        else layer_kwargs_dict['last'] if 'last' in layer_kwargs_dict and ii==Nlayers \
                                        else layer_kwargs))
                           for ii in range(Nlayers+1)])
    #}}}


    @staticmethod
    def __remove_layer_norm(d, key) :
        """
        helper function to remove layer norm from the layer_kwargs_dict at specific keys
        """
    #{{{
        if isinstance(key, int) :
            key = str(key)

        if key in d and 'layernorm' in d[key] :
            d[key].pop('layernorm')
    #}}}
