import torch
import torch.nn as nn

from network_encoder import NetworkEncoder
from network_decoder import NetworkDecoder
from spherical_model import SphericalModel
from data_batch import DataBatch


class Network(nn.Module) :
    """
    the entire network
    """

    def __init__(self) :
    #{{{
        super().__init__()
        k_latent = cfg.NETWORK_DEFAULT_NLATENT
        self.encoder = NetworkEncoder(k_latent)
        self.decoder = NetworkDecoder(k_latent, k_out=cfg.OUTPUT_NFEATURES)
        self.spherical = SphericalModel()
    #}}}

    
    def forward(self, batch) :
    #{{{ 
        assert isinstance(batch, DataBatch)

        x = self.encoder(batch.DM_coords, u=batch.u,
                         basis=batch.basis if cfg.NBASIS != 0 else None)
        x = self.decoder(x, batch.TNG_coords, u=batch.u,
                         basis=batch.basis if cfg.NBASIS != 0 else None,
                         # do not have an activation function before the final output
                         # since we generally want to map to the entire real line
                         layer_kwargs_dict=dict(last={'activation' : False})
        spherical = self.spherical(batch.M200c, batch.TNG_radii,
                                   R200c=batch.R200c if not cfg.NORMALIZE_COORDS else None)
        return Network.__combine(x, spherical)
    #}}}


    @staticmethod
    def __combine(x, spherical) :
        """
        combines the network output with the spherically symmetric prediction
        TODO this is something we can play with, e.g. multiply vs add, different functions applied to x, etc

        x ... output of the network, of shape [batch, Nvecs, Nfeatures],
              or a list of length batch of shapes [1, Nvecsi, Nfeatures]
        spherical ... the spherically symmetric simplified model of shape [batch, Nvecs, 1]
                      or a list of length batch of shapes [1, Nvecsi, 1]
        """
    #{{{
        if isinstance(x, list) :
            return [Network.__combine(xi, spherical[ii]) for ii, xi in enumerate(x)]

        return spherical + torch.exp(x)
    #}}}
