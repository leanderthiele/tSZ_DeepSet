import torch
import torch.nn as nn

from network_encoder import NetworkEncoder
from network_decoder import NetworkDecoder
from network_origin import NetworkOrigin
from network_deformer import NetworkDeformer
from network_batt12 import NetworkBatt12
from global_fields import GlobalFields
from basis import Basis
from data_batch import DataBatch
import cfg


class Network(nn.Module) :
    """
    the entire network
    """

    def __init__(self) :
    #{{{
        super().__init__()
        
        if cfg.NET_ARCH['enc_dec'] :
            k_latent = cfg.NETWORK_DEFAULT_NLATENT
            self.encoder = NetworkEncoder(k_latent)
            self.decoder = NetworkDecoder(k_latent, k_out=cfg.OUTPUT_NFEATURES,
                                          # do not have an activation function before the final output
                                          # since we generally want to map to the entire real line
                                          layer_kwargs_dict=dict(last={'activation' : False,
                                                                       'dropout' : None}))

        if cfg.NET_ARCH['origin'] :
            self.origin = NetworkOrigin()

        if cfg.NET_ARCH['deformer'] :
            self.deformer = NetworkDeformer()
        
        self.batt12 = NetworkBatt12()
    #}}}

    
    def forward(self, batch) :
    #{{{ 
        assert isinstance(batch, DataBatch)

        u = batch.u if len(GlobalFields) != 0 else None
        basis = batch.basis if len(GlobalFields) != 0 else None

        if cfg.NET_ARCH['origin'] :
            # first find the shifted origin
            o = self.origin(batch.DM_coords, u=u, basis=basis)

            # shift all coordinates according to this new origin
            batch = batch.add_origin(o)

        if cfg.NET_ARCH['enc_dec'] :
            # encode the DM field
            x = self.encoder(batch.DM_coords, u=u, basis=basis)

            # decode at the TNG particle positions
            x = self.decoder(x, batch.TNG_coords, r=batch.TNG_radii, u=u, basis=basis)

        if cfg.NET_ARCH['deformer'] :
            # now deform the TNG positions -- TODO experiment with the order of the two statements
            batch.TNG_radii = self.deformer(batch.TNG_coords, batch.TNG_radii, u=u, basis=basis)

        # now evaluate the (modified) Battaglia+2012 model at the deformed radial coordinates
        spherical = self.batt12(batch.M200c, batch.TNG_radii,
                                R200c=batch.R200c if not cfg.NORMALIZE_COORDS else None)

        if cfg.NET_ARCH['enc_dec'] :
            x = Network.__combine(x, spherical)
        else :
            x = spherical

        return x
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

        return spherical + x
    #}}}


    def to_device(self) :
    #{{{ 
        if cfg.DEVICE_IDX is not None :
            return self.to(cfg.DEVICE_IDX)
        else :
            return self
    #}}}
