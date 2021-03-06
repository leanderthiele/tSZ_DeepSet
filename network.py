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
from data_modes import DataModes
import cfg


class Network(nn.Module) :
    """
    the entire network
    """

    def __init__(self) :
    #{{{
        super().__init__()

        assert cfg.NET_ARCH['local'] or cfg.NET_ARCH['decoder'] or cfg.NET_ARCH['batt12']

        # make sure there were no typos when modifying the NET_ARCH dict
        assert set(cfg.NET_ARCH.keys()) == {'origin', 'batt12', 'deformer', 'encoder',
                                            'scalarencoder', 'decoder', 'vae', 'local', }
 
        if cfg.NET_ARCH['decoder'] :
 
            if cfg.NET_ARCH['encoder'] :
                self.encoder = NetworkEncoder(MLP_Nlayers=cfg.ENCODER_MLP_NLAYERS,
                                              MLP_Nhidden=cfg.ENCODER_MLP_NHIDDEN,
                                              MLP_kwargs_dict=dict(\
                                                last=dict(layer_kwargs_dict=dict(\
                                                   last={'bias_init': 'zeros_(%s)'}))))
            
            if cfg.NET_ARCH['scalarencoder'] :
                self.scalarencoder = NetworkScalarEncoder(MLP_Nlayers=cfg.SCALAR_ENCODER_MLP_NLAYERS,
                                                          MLP_Nhidden=cfg.SCALAR_ENCODER_MLP_NHIDDEN,
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

            self.decoder = NetworkDecoder(self.encoder.Nlatent if cfg.NET_ARCH['encoder'] else 0,
                                          self.scalarencoder.Nlatent if cfg.NET_ARCH['scalarencoder'] else 0,
                                          k_out=cfg.OUTPUT_NFEATURES,
                                          MLP_Nlayers=cfg.DECODER_DEFAULT_NLAYERS,
                                          MLP_Nhidden=cfg.DECODER_DEFAULT_NHIDDEN,
                                          dropout=cfg.DECODER_DROPOUT,
                                          visible_dropout=cfg.DECODER_VISIBLE_DROPOUT,
                                          # do not have an activation function before the final output
                                          # since we generally want to map to the entire real line
                                          # we also initialize the last bias to zero because our base model
                                          # expects zero output on average here
                                          layer_kwargs_dict=dict(last={'activation': False,
                                                                       'dropout': None,
                                                                       'bias_init': 'zeros_(%s)'}))

        else :

            # no decoder provided, this is unusual and we should check!
            assert not (cfg.NET_ARCH['encoder'] or cfg.NET_ARCH['scalarencoder'] or cfg.NET_ARCH['vae'])

            if cfg.NET_ARCH['local'] :
                assert cfg.LOCAL_NLATENT == cfg.OUTPUT_NFEATURES
                # if we have zero hidden layers, we can't concat with the number of particles
                # and still get something useful as output
                assert cfg.LOCAL_NLAYERS > 0

        if cfg.NET_ARCH['local'] :
            self.local = NetworkLocal(MLP_Nlayers=cfg.LOCAL_MLP_NLAYERS,
                                      MLP_Nhidden=cfg.LOCAL_MLP_NHIDDEN,
                                      dropout=None, # we'll always have enough training samples here
                                      MLP_kwargs_dict=dict(\
                                        last=dict(layer_kwargs_dict=dict(\
                                           last={'layernorm': False if not cfg.NET_ARCH['decoder'] else cfg.LAYERNORM,
                                                 'activation': False,
                                                 'bias_init': 'zeros_(%s)'}))))

        if cfg.NET_ARCH['origin'] :
            self.origin = NetworkOrigin(MLP_Nlayers=cfg.ORIGIN_MLP_NLAYERS,
                                        MLP_Nhidden=cfg.ORIGIN_MLP_NHIDDEN)

        if cfg.NET_ARCH['batt12'] :
            self.batt12 = NetworkBatt12(xc_fixed=cfg.NET_ARCH['deformer'])
            if cfg.NET_ARCH['deformer'] :
                self.deformer = NetworkDeformer(MLP_Nlayers=cfg.DEFORMER_NLAYERS,
                                                MLP_Nhidden=cfg.DEFORMER_NHIDDEN,
                                                dropout=cfg.DEFORMER_DROPOUT,
                                                visible_dropout=cfg.DEFORMER_VISIBLE_DROPOUT,
                                                layer_kwargs_dict=dict(last={'activation': False,
                                                                             'dropout': None,
                                                                             'bias_init': 'zeros_(%s)'}))

        if cfg.NET_ARCH['deformer'] :
            assert cfg.NET_ARCH['batt12']

        if (cfg.NET_ARCH['decoder'] or cfg.NET_ARCH['local']) and not cfg.NET_ARCH['batt12'] :
            assert cfg.OUTPUT_NFEATURES == 1

        if cfg.NET_ARCH['decoder'] or cfg.NET_ARCH['local'] :
            # register degree of freedom to rescale the network output
            self.register_parameter('scaling', nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))

        # validate that we didn't make a mistake
        assert all(v == hasattr(self, k) for k, v in cfg.NET_ARCH.items())
    #}}}

 
    def forward(self, batch, recon_seed=None, gauss_seeds=None) :
        """
        batch ... DataBatch instance holding all the data
        Following only apply if cfg.NET_ARCH['vae'] :
        recon_seed  ... single integer, can be used to deterministically set the RNG
                        in VAE reconstruction
        gauss_seeds ... list of integers, to generate several Gaussian outputs
                        or a single integer denoting the number of Gaussian outputs

        Returns:
            recon_prediction  ... the reconstructed prediction (tensor)
            gauss_predictions ... the Gaussian predictions (list)
            KLD               ... negative KL divergence
        """
    #{{{ 
        assert isinstance(batch, DataBatch)

        if gauss_seeds is not None :
            assert batch.mode is not DataModes.TRAINING

        if cfg.NET_ARCH['local'] :
            # NOTE this has to come before the origin net since DM and TNG particles
            #      may be shifted differently depending on cfg.ORIGIN_SHIFT_DM
            l = self.local(batch.TNG_coords, batch.DM_coords_local,
                           batch.DM_N_local, batch.basis, batch.DM_vels_local,
                           batch.P200c)
        else :
            l = None

        if cfg.NET_ARCH['origin'] :
            # first find the shifted origin, shape [batch, 2, 3]
            o_norm = self.origin(batch.DM_coords, v=batch.DM_vels, u=batch.u, basis=batch.basis)

            # we give the network the freedom to generally choose a mixture of multiplying by an
            # appropriate scale or simply taking a pre-determined vector
            # o is of shape [batch, 1, 3]
            o = (o_norm[:, 0, :] * (batch.Xoff / batch.R200c).unsqueeze(1).expand(-1, 3) \
                 + o_norm[:, 1, :]).unsqueeze(1)

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
            e = self.encoder(batch.DM_coords, v=batch.DM_vels, u=batch.u, basis=batch.basis)

        if cfg.NET_ARCH['scalarencoder'] :
            s = self.scalarencoder(batch.DM_coords, v=batch.DM_vels,
                                   u=batch.u if self.scalarencoder.globals_passed else None,
                                   basis=batch.basis if self.scalarencoder.basis_passed else None)

        if cfg.NET_ARCH['vae'] :
            z, KLD = self.vae(batch.TNG_residuals,
                              mode='code' if batch.mode is DataModes.TRAINING else 'mean',
                              seed=recon_seed)
            if gauss_seeds is not None :
                assert batch.mode is not DataModes.TRAINING

                if isinstance(gauss_seeds, list) :
                    z_gauss = [self.vae(batch.TNG_residuals, mode='gaussian', seed=seed) for seed in gauss_seeds]
                else :
                    z_gauss = [self.vae(batch.TNG_residuals, mode='gaussian') for _ in range(gauss_seeds)]
            else :
                z_gauss = None
        else :
            KLD = torch.zeros(len(batch))
            z_gauss = None

        if cfg.NET_ARCH['batt12'] :
            if cfg.NET_ARCH['deformer'] :
                # now deform the TNG positions -- TODO experiment with the order
                # relative to the decoder evaluation
                r_b12 = self.deformer(batch.TNG_coords, batch.TNG_radii,
                                      u=batch.u if self.deformer.globals_passed else None,
                                      basis=batch.basis)
            else :
                r_b12 = batch.TNG_radii

            # now evaluate the (modified) Battaglia+2012 model at the deformed radial coordinates
            b12 = self.batt12(batch.M200c, r_b12, batch.P200c)
        else :
            b12 = None

        if cfg.NET_ARCH['decoder'] :
            # decode at the TNG particle positions
            decoder_kwargs = {'x': batch.TNG_coords,
                              'h': e if cfg.NET_ARCH['encoder'] else None,
                              's': s if cfg.NET_ARCH['scalarencoder'] else None,
                              'r': batch.TNG_radii if self.decoder.r_passed else None,
                              'u': batch.u if self.decoder.globals_passed else None,
                              'basis': batch.basis if self.decoder.basis_passed else None,
                              'local': l if self.decoder.local_passed else None}
            x = self.decoder(vae=z if cfg.NET_ARCH['vae'] else None, **decoder_kwargs)

            if z_gauss is not None :
                x_gauss = [self.decoder(vae=zi, **decoder_kwargs) for zi in z_gauss]
            else :
                x_gauss = None
        else :
            x = None
            x_gauss=None

        f = self.__output(x, l, b12)

        if x_gauss is not None :
            f_gauss = [self.__output(xi, l, b12) for xi in x_gauss]
        else :
            f_gauss = None

        return f, f_gauss, KLD
    #}}}


    def __output(self, x, l, b12) :
        """
        Computes the final output from
            x   ... decoder output
            l   ... local output
            b12 ... Batt12 output
        """
    #{{{
        if cfg.NET_ARCH['decoder'] and not cfg.NET_ARCH['batt12'] :
            # TODO why not use exp here?
            return self.scaling * torch.relu(torch.sinh(x))
        elif cfg.NET_ARCH['decoder'] and cfg.NET_ARCH['batt12'] :
            return self.__combine(x, b12)
        elif not cfg.NET_ARCH['decoder'] and not cfg.NET_ARCH['local'] and cfg.NET_ARCH['batt12'] :
            return b12
        elif not cfg.NET_ARCH['decoder'] and cfg.NET_ARCH['local'] and cfg.NET_ARCH['batt12'] :
            return self.__combine(l, b12)
        elif not cfg.NET_ARCH['decoder'] and cfg.NET_ARCH['local'] and not cfg.NET_ARCH['batt12'] :
            # TODO why not use exp here?
            return self.scaling * torch.relu(torch.sinh(l))
        else :
            raise RuntimeError('Should not happen!')
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
            x = torch.relu(b12 + self.scaling * torch.sinh(x))
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
