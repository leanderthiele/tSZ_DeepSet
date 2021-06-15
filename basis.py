import numpy as np
import numpy.linalg as LA

from halo import Halo
from fixed_len_vec import FixedLenVec


class Basis(np.ndarray, metaclass=FixedLenVec) :
    """
    a wrapper around the numpy.ndarray, constructible from a Halo instance
    bundles the global vectorial features of the halo we want to use into a numpy array
    It is a principled idea to return the unit vectors only, the magnitudes can be passed
    through the global fields.
    """

    def __new__(cls, halo) :
        """
        constructing ndarray's is a bit tricky, so we do it this way
        """
    #{{{    
        assert isinstance(halo, Halo)

        ang_mom_unit = halo.ang_momentum_DM / LA.norm(halo.ang_momentum_DM)

        w, v = LA.eigh(halo.inertia_DM)
        v = v.T

        projections = v @ ang_mom_unit

        # choose the octant of the coordinate system where the angular momentum points
        # this fixes the coordinate system uniquely
        v[projections < 0] *= -1

        # central CM unit vectors
        central_CM_unit = halo.central_CM_DM / LA.norm(halo.central_CM_DM, axis=-1, keepdims=True)

        return np.concatenate([ang_mom_unit[None,:], v, central_CM_unit], axis=0).view(type=cls)
    #}}}

    
    @classmethod
    def _length(cls) :
        """
        should not be used directly, adapt if more features are added to the global vector
        """
    #{{{ 
        return 7
    #}}}
