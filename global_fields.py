import numpy as np

from halo import Halo
from fixed_len_vec import FixedLenVec


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

        return np.array([np.log(halo.M200c_DM), ]).view(type=cls)
    #}}}


    @classmethod
    def _length(cls) :
        """
        should not be used directly, adapt if more features are added to the global vector
        """
    #{{{
        return 1
    #}}}
