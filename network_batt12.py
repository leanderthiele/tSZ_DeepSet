import torch
import torch.nn as nn

import cfg


class NetworkBatt12(nn.Module) :
    """
    implements a version of the Battaglia+2012 profile with learnable parameters.
    We use the output from this model as the baseline, the actual DeepSet
    is then supposed to learn local corrections to this model.

    xc_fixed ... can keep the radial scale fixed, useful in order to remove degeneracy
                 with the deformer
    """

    def __init__(self, xc_fixed=False) :
    #{{{
        super().__init__()

        scalar_param = lambda x : nn.Parameter(torch.tensor(x, dtype=torch.float32))

        # this is what Battaglia+2012 fit for
        # we take the default values not from the B12 paper but from a simple fit using only the B12 model
        # NOTE these have been updated to TNG_RESOLUTION=64 (about 10% improvement)
        self.register_parameter('A_P0', scalar_param(3.9183))
        self.register_parameter('am_P0', scalar_param(0.5705))
        
        self.register_parameter('A_xc', scalar_param(2.8859))
        self.register_parameter('am_xc', scalar_param(-0.8130))

        self.register_parameter('A_beta', scalar_param(13.8758))
        self.register_parameter('am_beta', scalar_param(-0.6282))

        self.xc_fixed = xc_fixed
    #}}}
    

    def forward(self, M200c, r, P200c) :
        """
        M200c ... shape [batch]
        r     ... shape [batch, Nvecs, 1] or list of length batch with shapes [1, Nvecsi, 1]
                  (assumed normalized by R200c)
        P200c ... shape [batch]

        Returns thermal pressure at r, in the same shape as r
        (units depend on cfg.SCALE_PTH)
        """
    #{{{
        # do this here so no interference with InitModel
        if self.xc_fixed :
            self.A_xc.requires_grad = False
            self.am_xc.requires_grad = False

        if isinstance(r, list) :
            return [self(torch.tensor([M200c[ii],], requires_grad=False), ri)
                    for ii, ri in enumerate(r)]

        P0 = self.__primitive(M200c, 'P0')
        xc = self.__primitive(M200c, 'xc')
        beta = self.__primitive(M200c, 'beta')

        r = r / xc[:,None,None]

        # make sure we don't have a divergence if a particle is directly at the origin
        r += 1e-3

        return (1 if cfg.SCALE_PTH else P200c[:, None, None]) \
               * P0[:, None, None] * r.pow(-0.3) * ( 1 + r ).pow(-beta[:, None, None])
    #}}}


    def __primitive(self, M200c, s) :
        """
        computes the repeatedly used combination
        A * (M200c / 1e14 Msun)**alpha_m
        """
    #{{{
        # NOTE h is not really the Illustris h here, fitting parameters will absorbe
        return getattr(self, 'A_'+s) * (0.7 * M200c / 1e4).pow(getattr(self, 'am_'+s))
    #}}}


    def to_device(self) :
    #{{{
        if cfg.device_idx is not None :
            return self.to(cfg.device_idx)
        else :
            return self
    #}}}
