import numpy as np

from halo import Halo
from fixed_len_vec import FixedLenVec


class Basis(np.ndarray, metaclass=FixedLenVec) :
    """
    a wrapper around the numpy.ndarray, constructible from a Halo instance
    bundles the global vectorial features of the halo we want to use into a numpy array
    """

    def __new__(cls, halo) :
        """
        constructing ndarray's is a bit tricky, so we do it this way
        """
    #{{{    
        assert isinstance(halo, Halo)

        # FIXME
        return None
    #}}}

    
    @classmethod
    def _length(cls) :
        """
        should not be used directly, adapt if more features are added to the global vector
        """
    #{{{ 
        return 0
    #}}}
