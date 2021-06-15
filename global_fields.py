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

        logM = np.log(halo.M200c_DM)
        ang_mom_norm = LA.norm(halo.ang_momentum_DM)

        w, v = LA.eigh(halo.inertia_DM)
        v = v.T # get into better format -- v[i, :] is the ith eigenvector

        # compute the angles between angular momentum and inertia axes
        # note that the eigenvectors returned by eigh are already normalized
        angles = np.arccos(v @ halo.ang_momentum_DM / ang_mom_norm)

        # compute the central CM magnitudes
        central_CM_norm = LA.norm(halo.central_CM, axis=-1)

        # map to the 0, pi/2 interval -- mods out the symmetry under parity
        # (np.arccos returns in the interval [0, pi])
        for ii in range(3) :
            if angles[ii] > 0.5*np.pi :
                angles[ii] = np.pi - angles[ii]

        if cfg.NORMALIZE_COORDS :
            ang_mom_norm *= cfg.UNIT_MASS / (halo.R200c_DM * halo.V200c_DM * halo.M200c_DM)
            w *= cfg.UNIT_MASS / (halo.R200c_DM**2 * halo.M200c_DM)

        out = np.array([logM, ang_mom_norm, *w, *angles, *central_CM_norm]).view(type=cls)

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

        return out
    #}}}


    @classmethod
    def _length(cls) :
        """
        should not be used directly, adapt if more features are added to the global vector
        """
    #{{{
        return 11
    #}}}
