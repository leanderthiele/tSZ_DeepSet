import numpy as np

from halo import Halo
from global_fields import GlobalFields
import cfg

class HaloCatalog :

    def __init__(self, indices) :
        """
        halo_catalog ... a dict with the fields of halo_catalog.npz and some additional ones
        indices      ... the indices to pass to the Halo constructors
        """
    #{{{
        halo_catalog = dict(np.load(cfg.HALO_CATALOG))

        self.halos = []
        
        if len(GlobalFields) != 0 :
            # these simply hold a numpy array with the global features
            # of each halo
            u = []

        for idx in indices :
            self.halos.append(Halo(halo_catalog, idx))
            if len(GlobalFields) != 0 :
                u.append(GlobalFields(self.halos[-1]))
                u = np.array(u)

        if len(GlobalFields) != 0 :
            dglobals = np.empty((len(self.halos), len(GlobalFields)))
            
            # ok, this is probably not efficient but who cares, these arrays are small
            # and we do this O(1) times
            for ii in range(len(self.halos)) :
                
                dglobals = np.empty(len(GlobalFields))

                for jj in range(len(GlobalFields)) :
                    sorted_diffs = np.sort(np.fabs(u[:,jj] - u[ii,jj]))
                    assert sorted_diffs[0] < 1e-5
                    dglobals[jj] = sorted_diffs[1]

               self.halos[ii].dglobals = dglobals 
    #}}}

    
    def __len__(self) :
    #{{{
        return len(self.halos)
    #}}}


    def __getitem__(self, idx) :
    #{{{
        if idx >= len(self) or idx < 0 :
            raise IndexError

        return self.halos[idx]
    #}}}
