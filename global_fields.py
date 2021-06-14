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

    def __new__(cls, halo) :
        """
        constructing ndarray's is a bit tricky, so we do it this way
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

        # map to the 0, pi/2 interval -- mods out the symmetry under parity
        # (np.arccos returns in the interval [0, pi])
        for ii in range(3) :
            if angles[ii] > 0.5*np.pi :
                angles[ii] = np.pi - angles[ii]

        if cfg.NORMALIZE_COORDS :
            ang_mom_norm *= cfg.UNIT_MASS / (halo.R200c_DM * halo.V200c_DM * halo.M200c_DM)
            w *= cfg.UNIT_MASS / (halo.R200c_DM**2 * halo.M200c_DM)

        return np.array([logM, ang_mom_norm, *w, *angles]).view(type=cls)
    #}}}


    @classmethod
    def _length(cls) :
        """
        should not be used directly, adapt if more features are added to the global vector
        """
    #{{{
        return 8
    #}}}
