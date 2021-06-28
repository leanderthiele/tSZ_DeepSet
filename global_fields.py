import numpy as np
import numpy.linalg as LA

from halo import Halo
from fixed_len_vec import FixedLenVec
import cfg


class GlobalFields(np.ndarray, metaclass=FixedLenVec) :
    """
    a wrapper around the numpy.ndarray, constructible from a Halo instance
    bundles the global scalar features of the halo we want to use into a numpy array
    """

    def __new__(cls, halo, rng=None) :
        """
        constructing ndarray's is a bit tricky, so we do it this way
        halo ... the halo for which the global values are to be extracted
        rng  ... if passed, should be a numpy random number generator instance.
                 Using halo.dglobals as characteristic scale, this will be used
                 to add some noise to the global features and hopefully mitigate
                 overfitting.
        """
    #{{{
        assert isinstance(halo, Halo)

        # logarithmic mass
        logM = np.log(halo.M200c)

        # magnitude of angular momentum vector
        ang_mom_norm = LA.norm(halo.ang_momentum)

        # relaxation measures
        Xoff = halo.Xoff
        Voff = halo.Voff

        # diagonalize inertia tensor
        eigval_inertia, eigvec_inertia = LA.eigh(halo.inertia)
        eigvec_inertia = eigvec_inertia.T # get into better format -- v[i, :] is the ith eigenvector

        # compute the angles between angular momentum and inertia axes
        # note that the eigenvectors returned by eigh are already normalized
        inertia_dot_ang_mom = eigvec_inertia @ halo.ang_momentum / ang_mom_norm

        # diagonalize velocity dispersion tensor
        eigval_vel_dispersion, eigvec_vel_dispersion = LA.eigh(halo.vel_dispersion)
        eigvec_vel_dispersion = eigvec_vel_dispersion.T

        # compute the angles between angular momentum and velocity dispersion axes
        vel_dispersion_dot_ang_mom = eigvec_vel_dispersion @ halo.ang_momentum / ang_mom_norm

        # compute angles between inertia and velocity dispersion eigenvectors
        vel_dispersion_dot_inertia = np.einsum('id,jd->ij', eigvec_inertia, eigvec_vel_dispersion).flatten()

        # fix the parity symmetry
        inertia_dot_ang_mom[inertia_dot_ang_mom < 0] *= -1
        vel_dispersion_dot_ang_mom[vel_dispersion_dot_ang_mom < 0] *= -1
        vel_dispersion_dot_inertia[vel_dispersion_dot_inertia < 0] *= -1

        if cfg.NORMALIZE_COORDS :
            ang_mom_norm /= halo.R200c * halo.V200c * halo.M200c
            Xoff /= halo.R200c
            Voff /= halo.V200c
            eigval_inertia *= cfg.UNIT_MASS / (halo.R200c**2 * halo.M200c)
            eigval_vel_dispersion *= cfg.UNIT_MASS / (halo.V200c**2 * halo.M200c)

        out = []

        if cfg.GLOBALS_USE['logM'] :
            out.append(logM)
        if cfg.GLOBALS_USE['ang_mom'] :
            out.append(ang_mom_norm)
        if cfg.GLOBALS_USE['Xoff'] :
            out.append(Xoff)
        if cfg.GLOBALS_USE['Voff'] :
            out.append(Voff)
        if cfg.GLOBALS_USE['inertia'] :
            out.extend(eigval_inertia)
        if cfg.GLOBALS_USE['inertia_dot_ang_mom'] :
            out.extend(inertia_dot_ang_mom)
        if cfg.GLOBALS_USE['vel_dispersion'] :
            out.extend(eigval_vel_dispersion)
        if cfg.GLOBALS_USE['vel_dispersion_dot_ang_mom'] :
            out.extend(vel_dispersion_dot_ang_mom)
        if cfg.GLOBALS_USE['vel_dispersion_dot_inertia'] :
            out.extend(vel_dispersion_dot_inertia)

        out = np.array(out)

        # add noise if requested
        if rng is not None and cfg.GLOBALS_NOISE is not None :

            assert np.all(halo.dglobals > 0)

            for ii in range(len(out)) :

                # generate Gaussian random number of unit variance and zero mean
                r = rng.normal()

                # rescale by the dglobals entry
                r *= halo.dglobals[ii,0] if r<0 else halo.dglobals[ii,1]

                # add to the output with appropriate weight
                out[ii] += r * cfg.GLOBALS_NOISE

        # if halo has u_avg and u_std fields populated, normalize the output
        # (NOTE that the order is important here, we only normalize *after* adding the noise)
        if hasattr(halo, 'u_avg') and hasattr(halo, 'u_std') :
            out = (out - halo.u_avg) / halo.u_std

        return out.view(type=cls)
    #}}}


    @classmethod
    def _length(cls) :
        """
        should not be used directly
        """
    #{{{
        return (not cfg.GLOBALS_USE['none']) * (cfg.GLOBALS_USE['logM']
                                                + cfg.GLOBALS_USE['ang_mom']
                                                + cfg.GLOBALS_USE['Xoff']
                                                + cfg.GLOBALS_USE['Voff']
                                                + 3 * cfg.GLOBALS_USE['inertia']
                                                + 3 * cfg.GLOBALS_USE['inertia_dot_ang_mom'] 
                                                + 3 * cfg.GLOBALS_USE['vel_dispersion']
                                                + 3 * cfg.GLOBALS_USE['vel_dispersion_dot_ang_mom']
                                                + 9 * cfg.GLOBALS_USE['vel_dispersion_dot_inertia'])
    #}}}
