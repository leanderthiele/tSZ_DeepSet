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
        self.register_parameter('A_P0', scalar_param(18.1))
        self.register_parameter('am_P0', scalar_param(0.154))
        self.register_parameter('A_xc', scalar_param(0.497))
        self.register_parameter('am_xc', scalar_param(-0.00865))
        self.register_parameter('A_beta', scalar_param(4.35))
        self.register_parameter('am_beta', scalar_param(0.0393))

        if xc_fixed :
            self.A_xc.requires_grad = False
            self.am_xc.requires_grad = False
    #}}}
    

    def forward(self, M200c, r, R200c=None) :
        """
        M200c ... shape [batch]
        r     ... shape [batch, Nvecs, 1] or list of length batch with shapes [1, Nvecsi, 1]

        Returns thermal pressure at r in units of P200c, in the same shape as r
        if R200c is not None, we assume that the radii are not normalized yet
        (then it should be of shape [batch]
        """
    #{{{
        if isinstance(r, list) :
            return [self(torch.tensor([M200c[ii],], requires_grad=False), ri,
                         R200c=None if R200c is None else torch.tensor([R200c[ii], ], requires_grad=False))
                    for ii, ri in enumerate(r)]

        if R200c is not None :
            assert not cfg.NORMALIZE_COORDS
            r /= R200c[:,None,None]

        P0 = self.__primitive(M200c, 'P0')
        xc = self.__primitive(M200c, 'xc')
        beta = self.__primitive(M200c, 'beta')

        r = r / xc[:,None,None]

        # make sure we don't have a divergence if a particle is directly at the origin
        r += 1e-3

        return P0[:,None,None] * r.pow(-0.3) * ( 1 + r ).pow(-beta[:,None,None])
    #}}}


    def __primitive(self, M200c, s) :
        """
        computes the repeatedly used combination
        A * (M200c / 1e14 Msun)**alpha_m
        """
    #{{{
        return getattr(self, 'A_'+s) * (0.7 * M200c / 1e4).pow(getattr(self, 'am_'+s))
    #}}}


    def to_device(self) :
    #{{{
        if cfg.device_idx is not None :
            return self.to(cfg.device_idx)
        else :
            return self
    #}}}
