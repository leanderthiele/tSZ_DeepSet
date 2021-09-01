import torch
import torch.nn as nn

from network_encoder import NetworkEncoder
from network_scalarencoder import NetworkScalarEncoder
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

        assert cfg.NET_ARCH['local'] or cfg.NET_ARCH['decoder'] or cfg.NET_ARCH['batt12']
 
        if cfg.NET_ARCH['decoder'] :
 
            if cfg.NET_ARCH['encoder'] :
                self.encoder = NetworkEncoder(MLP_Nlayers=cfg.ENCODER_MLP_NLAYERS,
                                              MLP_Nhidden=cfg.ENCODER_MLP_NHIDDEN)
            
            if cfg.NET_ARCH['scalarencoder'] :
                self.scalarencoder = NetworkScalarEncoder()

            self.decoder = NetworkDecoder(self.encoder.Nlatent if cfg.NET_ARCH['encoder'] else 0,
                                          self.scalarencoder.Nlatent if cfg.NET_ARCH['scalarencoder'] else 0,
                                          k_out=cfg.OUTPUT_NFEATURES,
                                          MLP_Nlayers=cfg.DECODER_DEFAULT_NLAYERS,
                                          MLP_Nhidden=cfg.DECODER_DEFAULT_NHIDDEN,
                                          # do not have an activation function before the final output
                                          # since we generally want to map to the entire real line
                                          # we also initialize the last bias to zero because our base model
                                          # expects zero output on average here
                                          layer_kwargs_dict=dict(last={'activation' : False,
                                                                       'dropout' : None,
                                                                       'bias_init': 'zeros_(%s)'}))

            # register degree of freedom to rescale the network output
            self.register_parameter('scaling', nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))

        else :
            # no decoder provided, this is unusual and we should check!
            assert not (cfg.NET_ARCH['encoder'] or cfg.NET_ARCH['scalarencoder'] or cfg.NET_ARCH['vae'])
            if cfg.NET_ARCH['local'] :
                assert cfg.LOCAL_NLATENT == cfg.OUTPUT_NFEATURES

        if cfg.NET_ARCH['local'] :
            self.local = NetworkLocal(MLP_Nlayers=cfg.LOCAL_MLP_NLAYERS,
                                      MLP_Nhidden=cfg.LOCAL_MLP_NHIDDEN,
                                      dropout=None, # we'll always have enough training samples here
                                      MLP_kwargs_dict=dict(\
                                        last=dict(layer_kwargs_dict=dict(\
                                           last={'bias_init': 'zeros_(%s)'}))))

        if cfg.NET_ARCH['vae'] :
            self.vae = NetworkVAE(MLP_Nlayers=cfg.VAE_NLAYERS,
                                  MLP_Nhidden=cfg.VAE_NHIDDEN,
                                  # apparently it is a bad idea to have dropout in VAE encoder,
                                  # which sort of makes sense
                                  dropout=None,
                                  layer_kwargs_dict=dict(last={'bias_init': 'zeros_(%s)'}))

        if cfg.NET_ARCH['origin'] :
            self.origin = NetworkOrigin(MLP_Nlayers=cfg.ORIGIN_MLP_NLAYERS,
                                        MLP_Nhidden=cfg.ORIGIN_MLP_NHIDDEN)

        if cfg.NET_ARCH['batt12'] :
            self.batt12 = NetworkBatt12(xc_fixed=cfg.NET_ARCH['deformer'])
            if cfg.NET_ARCH['deformer'] :
                self.deformer = NetworkDeformer()

        if cfg.NET_ARCH['deformer'] :
            assert cfg.NET_ARCH['batt12']

        if (cfg.NET_ARCH['decoder'] or cfg.NET_ARCH['local']) and not cfg.NET_ARCH['batt12'] :
            assert cfg.OUTPUT_NFEATURES == 1

        # validate that we didn't make a mistake
        assert all(v == hasattr(self, k) for k, v in cfg.NET_ARCH.items())
    #}}}

 
    def forward(self, batch) :
    #{{{ 
        assert isinstance(batch, DataBatch)

        if cfg.NET_ARCH['local'] :
            l = self.local(batch.TNG_coords, batch.DM_coords_local,
                           batch.DM_N_local, batch.basis, batch.DM_vels_local,
                           batch.P200c)

        if cfg.NET_ARCH['origin'] :
            # first find the shifted origin
            o_norm = self.origin(batch.DM_coords, v=batch.DM_vels, u=batch.u, basis=batch.basis)

            # we want the origin network to be nicely normalized, so multiply by an appropriate scale
            # o is of shape [batch, 1, 3]
            o = o_norm * (batch.Xoff / batch.R200c).unsqueeze(1).unsqueeze(1).expand(-1, -1, 3)

            # in order to stabilize training (and make some asserts safe), we should make sure
            # the origin is not shifted too much
            # We crop at 0.5*R200c in each coordinate direction, this is very conservative and
            # makes later asserts on radii safe
            # tanh(x) is pretty close to linear for |x| < 0.3, so this should be fine
            o = 0.5 * torch.tanh(o)

            # shift all coordinates according to this new origin
            batch = batch.add_origin(o)

        if cfg.NET_ARCH['encoder'] :
            # encode the DM field
            x = self.encoder(batch.DM_coords, v=batch.DM_vels, u=batch.u, basis=batch.basis)

        if cfg.NET_ARCH['scalarencoder'] :
            s = self.scalarencoder(batch.DM_coords, v=batch.DM_vels,
                                   u=batch.u if self.scalarencoder.globals_passed else None,
                                   basis=batch.basis if self.scalarencoder.basis_passed else None)

        if cfg.NET_ARCH['vae'] :
            z, KLD = self.vae(batch.TNG_residuals)
        else :
            KLD = torch.zeros(len(batch))

        if cfg.NET_ARCH['decoder'] :
            # decode at the TNG particle positions
            x = self.decoder(batch.TNG_coords,
                             h=x if cfg.NET_ARCH['encoder'] else None,
                             s=s if cfg.NET_ARCH['scalarencoder'] else None,
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
            b12 = self.batt12(batch.M200c, r_b12, batch.P200c)

        if cfg.NET_ARCH['decoder'] and not cfg.NET_ARCH['batt12'] :
            x = self.scaling * torch.relu(torch.sinh(x))
        elif cfg.NET_ARCH['decoder'] and cfg.NET_ARCH['batt12'] :
            x = self.__combine(x, b12)
        elif not cfg.NET_ARCH['decoder'] and not cfg.NET_ARCH['local'] and cfg.NET_ARCH['batt12'] :
            x = b12
        elif not cfg.NET_ARCH['decoder'] and cfg.NET_ARCH['local'] and cfg.NET_ARCH['batt12'] :
            x = self.__combine(l, b12)
        elif not cfg.NET_ARCH['decoder'] and cfg.NET_ARCH['local'] and not cfg.NET_ARCH['batt12'] :
            x = torch.relu(torch.sinh(l))
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
