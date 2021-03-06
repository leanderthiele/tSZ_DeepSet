import torch
import torch.nn as nn

from network_mlp import NetworkMLP
from global_fields import GlobalFields
from basis import Basis
import normalization
from default_from_cfg import DefaultFromCfg
import cfg


class NetworkDecoder(nn.Module) :
    """
    takes the latent space vectors and the positions where we want to evaluate
    and gives us scalar outputs at the requested locations
    (if requested, k_out can be chosen as non-unity to have multiple values at those
     locations, but they will no longer be vector valued!)
    """

    def __init__(self, Nvecs, Nscals, k_out=1,
                 r_passed=DefaultFromCfg('DECODER_DEFAULT_R_PASSED'),
                 basis_passed=DefaultFromCfg('DECODER_DEFAULT_BASIS_PASSED'),
                 globals_passed=DefaultFromCfg('DECODER_DEFAULT_GLOBALS_PASSED'),
                 local_passed=DefaultFromCfg('NET_ARCH["local"]'), 
                 vae_passed=DefaultFromCfg('NET_ARCH["vae"]'),
                 **MLP_kwargs) :
        """
        Nvecs      ... the number of latent space vectors (can be zero, then h should be None in forward)
        Nscals     ... the number of latent space scalars
        k_out      ... the number of features to predict at the locations
        r_passed   ... whether the TNG radial coordinates will be passed
        basis_passed   ... whether the basis vectors will be passed
        globals_passed ... whether the globals will be passed
        local_passed   ... whether the local scalars will be passed
        vae_passed ... whether latent space from VAE will be passed
        MLP_kwargs ... to specify the multi-layer perceptron used here
        """
    #{{{
        if isinstance(r_passed, DefaultFromCfg) :
            r_passed = r_passed()
        if isinstance(basis_passed, DefaultFromCfg) :
            basis_passed = basis_passed()
        if isinstance(globals_passed, DefaultFromCfg) :
            globals_passed = globals_passed()
        if isinstance(local_passed, DefaultFromCfg) :
            local_passed = local_passed()
        if isinstance(vae_passed, DefaultFromCfg) :
            vae_passed = vae_passed()

        super().__init__()

        self.mlp = NetworkMLP(3 * Nvecs # one for TNG coord projection, one for TNG-vec distance, one for modulus
                              + Nscals
                              + r_passed
                              + globals_passed * len(GlobalFields)
                              + basis_passed * len(Basis)
                              + local_passed * cfg.LOCAL_NLATENT
                              + vae_passed * cfg.VAE_NLATENT,
                              k_out, **MLP_kwargs)

        self.Nvecs = Nvecs
        self.Nscals = Nscals
        self.r_passed = r_passed
        self.globals_passed = globals_passed
        self.basis_passed = basis_passed
        self.local_passed = local_passed
        self.vae_passed = vae_passed
    #}}}


    def create_scalars(self, x, h=None, s=None, r=None, u=None, basis=None, local=None, vae=None) :
    #{{{
        scalars = None
        desc = 'input to NetworkDecoder: '

        N_TNG = x.shape[1]

        if r is not None :
            x_norm = r + 1e-5
        else :
            x_norm = torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-5

        # we know that the TNG positions are normalized by R200c,
        # but need to take the origin shift into account
        assert x_norm.max().item() < 3.5, x_norm.max().item()

        if h is not None :
            # compute the moduli of the latent vectors, shape [batch, Nfeatures, 1]
            h_norm = torch.linalg.norm(h, dim=-1, keepdim=True)

            # harden against reordering by future me
            assert scalars is None

            scalars = h_norm.unsqueeze(1).squeeze(dim=-1).expand(-1, N_TNG, -1).clone()
            desc += '|h| [%d]; '%self.Nvecs

            # compute the projections of shape [batch, Nvects, latent feature]
            # TODO whether to normalize to unit vectors here is a hyperparameter we should explore!!!
            scalars = torch.cat((scalars,
                                 torch.einsum('bvd,bld->bvl', x / x_norm, h / (h_norm + 1e-5))),
                                dim=-1)
            desc += 'x.h [%d]; '%self.Nvecs

            # compute the vector distances of shape [batch, Nvects, latent feature]
            scalars = torch.cat((scalars,
                                 torch.linalg.norm(x.unsqueeze(2)-h.unsqueeze(1), dim=-1)),
                                dim=-1)
            desc += '|x-h| [%d]; '%self.Nvecs

        if s is not None :
            # concatenate with the halo-scale scalars (note this is exactly identical to the u-scalars)
            s_expanded = s.unsqueeze(1).expand(-1, N_TNG, -1)
            scalars = torch.cat((scalars, s_expanded), dim=-1) \
                      if scalars is not None \
                      else s_expanded
            desc += 's [%d]'%self.Nscals

        # concatenate with the radial distances if requested
        if r is not None :
            assert self.r_passed
            r_normed = normalization.TNG_radii(r)
            scalars = torch.cat((scalars, r_normed), dim=-1) if scalars is not None \
                      else r_normed
            desc += '|x| [1]; '
        else :
            assert not self.r_passed

        # concatenate with the basis projections if requested
        if basis is not None :
            assert self.basis_passed and len(Basis) != 0
            basis_projections = normalization.unit_contraction(torch.einsum('bid,bnd->bin',
                                                                            x / x_norm,
                                                                            basis))
            scalars = torch.cat((scalars, basis_projections), dim=-1) if scalars is not None \
                      else basis_projections
            desc += 'x.basis [%d]; '%len(Basis)
        else :
            assert not self.basis_passed or len(Basis) == 0

        # concatenate with the global vector if requested
        if u is not None :
            assert self.globals_passed and len(GlobalFields) != 0
            u_expanded = u.unsqueeze(1).expand(-1, N_TNG, -1)
            scalars = torch.cat((scalars, u_expanded), dim=-1) \
                      if scalars is not None \
                      else u_expanded
            desc += 'u [%d]; '%len(GlobalFields)
        else :
            assert not self.globals_passed or len(GlobalFields) == 0

        # concatenate with the local latent variables if requested
        if local is not None :
            assert self.local_passed
            scalars = torch.cat((scalars, local), dim=-1) \
                      if scalars is not None \
                      else local
            desc += 'local [%d]; '%cfg.LOCAL_NLATENT
        else :
            assert not self.local_passed

        # concatenate with the VAE latent variables if requested
        if vae is not None :
            assert self.vae_passed
            vae_expanded = vae.unsqueeze(1).expand(-1, N_TNG, -1)
            scalars = torch.cat((scalars, vae_expanded), dim=-1) \
                      if scalars is not None \
                      else vae_expanded
            desc += 'vae [%d]; '%cfg.VAE_NLATENT
        else :
            assert not self.vae_passed

        return scalars, desc
    #}}}


    def forward(self, x, h=None, s=None, r=None, u=None, basis=None, local=None, vae=None) :
        """
        x ... the positions where to evaluate, of shape [batch, Nvects, 3]
              or a list of length batch and shapes [1, Nvectsi, 3]
        h ... the latent vectors, of shape [batch, latent feature, 3]
        s ... the latent scalars, of shape [batch, latent_feature]
        r ... the radial positions where to evaluate, of shape [batch, Nvects, 1]
              or a list of length batch shapes [1, Nvectsi, 1]
        u ... the global vector, of shape [batch, Nglobals]
        basis ... the basis vectors to use -- either None if no basis is provided
                  or of shape [batch, Nbasis, 3]
        local ... the latent representation of the local environment
                  -- either None if no local DeepSet is used or of shape
                     [batch, Nvects, Nfeatures] or a list of length batch and shapes [1, Nvectsi, Nfeatures]
        vae ... the latent representation in the vae -- either None if no VAE is used
                or of shape [batch, Nlatent]
        """
    #{{{
        if isinstance(x, list) :
            return [self(xi,
                         h[ii, ...].unsqueeze(0),
                         r=r[ii] if r is not None else r,
                         u=u[ii, ...].unsqueeze(0) if u is not None else u,
                         basis=basis[ii, ...].unsqueeze(0) if basis is not None else basis,
                         local=local[ii] if local is not None else local,
                         vae=vae[ii, ...].unsqueeze(0) if vae is not None else vae)
                    for ii, xi in enumerate(x)]

        scalars, _ = self.create_scalars(x, h, s, r, u, basis, local, vae)

        # pass through the MLP, transform scalars -> scalars
        return self.mlp(scalars)
    #}}}
