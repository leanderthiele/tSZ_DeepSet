import torch
import torch.nn as nn

from network_encoder import NetworkEncoder
from network_decoder import NetworkDecoder
from network_origin import NetworkOrigin
from network_deformer import NetworkDeformer
from network_batt12 import NetworkBatt12
from network_local import NetworkLocal
from network_vae import NetworkVAE
from data_batch import DataBatch
import cfg


class Network(nn.Module) :
    """
    the entire network
    """

    def __init__(self) :
    #{{{
        super().__init__()

        assert cfg.NET_ARCH['decoder'] or cfg.NET_ARCH['batt12']
        
        if cfg.NET_ARCH['decoder'] :
            
            if cfg.NET_ARCH['encoder'] :
                k_latent = cfg.NETWORK_DEFAULT_NLATENT
                self.encoder = NetworkEncoder(k_latent)
            else :
                k_latent = 0

            self.decoder = NetworkDecoder(k_latent, k_out=cfg.OUTPUT_NFEATURES,
                                          MLP_Nlayers=cfg.DECODER_DEFAULT_NLAYERS,
                                          MLP_Nhidden=cfg.DECODER_DEFAULT_NHIDDEN,
                                          # do not have an activation function before the final output
                                          # since we generally want to map to the entire real line
                                          # we also initialize the last bias to zero because our base model
                                          # expects zero output on average here
                                          layer_kwargs_dict=dict(last={'activation' : False,
                                                                       'dropout' : None,
                                                                       'bias_init': 'zeros_(%s)'}))

        if cfg.NET_ARCH['local'] :
            assert cfg.NET_ARCH['decoder']
            self.local = NetworkLocal(MLP_kwargs_dict=dict(\
                last=dict(layer_kwargs_dict=dict(\
                    last={'bias_init': 'zeros_(%s)'}))))

        if cfg.NET_ARCH['vae'] :
            assert cfg.NET_ARCH['decoder']
            self.vae = NetworkVAE(MLP_Nlayers=cfg.VAE_NLAYERS, MLP_Nhidden=cfg.VAE_NHIDDEN,
                                  # apparently it is a bad idea to have dropout in VAE encoder,
                                  # which sort of makes sense
                                  dropout=None,
                                  layer_kwargs_dict=dict(last={'bias_init': 'zeros_(%s)'}))

        if cfg.NET_ARCH['encoder'] :
            assert cfg.NET_ARCH['decoder']

        if cfg.NET_ARCH['origin'] :
            self.origin = NetworkOrigin()

        if cfg.NET_ARCH['batt12'] :
            self.batt12 = NetworkBatt12(xc_fixed=cfg.NET_ARCH['deformer'])
            if cfg.NET_ARCH['deformer'] :
                self.deformer = NetworkDeformer()

        if cfg.NET_ARCH['deformer'] :
            assert cfg.NET_ARCH['batt12']

        if cfg.NET_ARCH['decoder'] :
            # register degree of freedom to rescale the network output
            self.register_parameter('scaling', nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))

        if cfg.NET_ARCH['decoder'] and not cfg.NET_ARCH['batt12'] :
            assert cfg.OUTPUT_NFEATURES == 1
    #}}}

    
    def forward(self, batch) :
    #{{{ 
        assert isinstance(batch, DataBatch)

        if cfg.NET_ARCH['local'] :
            l = self.local(batch.TNG_coords, batch.DM_coords_local,
                           batch.DM_N_local, batch.basis, batch.DM_vels_local)

        if cfg.NET_ARCH['origin'] :
            # first find the shifted origin
            o_norm = self.origin(batch.DM_coords, v=batch.DM_vels, u=batch.u, basis=batch.basis)

            # we want the origin network to be nicely normalized, so multiply by an appropriate scale
            # o is of shape [batch, 1, 3]
            o = o_norm * (batch.Xoff / batch.R200c).unsqueeze(1).unsqueeze(1).expand(-1, -1, 3)

            # shift all coordinates according to this new origin
            batch = batch.add_origin(o)

        if cfg.NET_ARCH['encoder'] :
            # encode the DM field
            x = self.encoder(batch.DM_coords, v=batch.DM_vels, u=batch.u, basis=batch.basis)

        if cfg.NET_ARCH['vae'] :
            z, KLD = self.vae(batch.TNG_residuals)
        else :
            KLD = torch.zeros(len(batch))

        if cfg.NET_ARCH['decoder'] :
            # decode at the TNG particle positions
            x = self.decoder(batch.TNG_coords,
                             h=x if cfg.NET_ARCH['encoder'] else None,
                             r=batch.TNG_radii if self.decoder.r_passed else None,
                             u=batch.u if self.decoder.globals_passed else None,
                             basis=batch.basis if self.decoder.basis_passed else None,
                             local=l if self.decoder.local_passed else None,
                             vae=z if cfg.NET_ARCH['vae'] else None)

        if cfg.NET_ARCH['batt12'] :
            if cfg.NET_ARCH['deformer'] :
                # now deform the TNG positions -- TODO experiment with the order
                # relative to the decoder evaluation
                r_b12 = self.deformer(batch.TNG_coords, batch.TNG_radii,
                                      u=batch.u, basis=batch.basis)
            else :
                r_b12 = batch.TNG_radii

            # now evaluate the (modified) Battaglia+2012 model at the deformed radial coordinates
            b12 = self.batt12(batch.M200c, r_b12)

        if cfg.NET_ARCH['decoder'] and not cfg.NET_ARCH['batt12'] :
            x = self.scaling * torch.relu(torch.sinh(x))
        elif cfg.NET_ARCH['decoder'] and cfg.NET_ARCH['batt12'] :
            x = self.__combine(x, b12)
        elif not cfg.NET_ARCH['decoder'] and cfg.NET_ARCH['batt12'] :
            x = b12
        else :
            raise RuntimeError('Should not happen!')

        return x, KLD
    #}}}


    def __combine(self, x, b12) :
        """
        combines the network output with the spherically symmetric prediction
        TODO this is something we can play with, e.g. multiply vs add, different functions applied to x, etc

        x ... output of the network, of shape [batch, Nvecs, Nfeatures],
              or a list of length batch of shapes [1, Nvecsi, Nfeatures]
        b12 ... the (modified) Battaglia+2012 simplified model of shape [batch, Nvecs, 1]
                or a list of length batch of shapes [1, Nvecsi, 1]
        """
    #{{{
        if isinstance(x, list) :
            return [self.__combine(xi, b12[ii]) for ii, xi in enumerate(x)]

        if cfg.OUTPUT_NFEATURES == 1 :
            x = b12 + self.scaling * torch.sinh(x)
        elif cfg.OUTPUT_NFEATURES == 2 :
            x = b12 * torch.relu(1 + x[..., 0].unsqueeze(-1)) \
                + self.scaling * torch.sinh(x[..., 1].unsqueeze(-1))
        else :
            raise RuntimeError(f'Invalid cfg.OUTPUT_NFEATURES: {cfg.OUTPUT_NFEATURES}')

        # map to positive real line
        return torch.relu(x)
    #}}}


    def to_device(self) :
    #{{{ 
        if cfg.device_idx is not None :
            return self.to(cfg.device_idx)

        return self
    #}}}
