import numpy as np

from data_modes import DataModes
from halo import Halo
from global_fields import GlobalFields
import cfg

class HaloCatalog :

    def __init__(self, mode) :
        """
        mode ... the mode for this halo catalog
        """
    #{{{
        assert isinstance(mode, DataModes)

        halo_catalog = dict(np.load(cfg.HALO_CATALOG))

        self.halos = []
        
        indices = mode.sample_indices()

        for idx in indices :
            self.halos.append(Halo(halo_catalog, idx))

        # if we are training and there are global fields, need to populate the dglobals
        # member variable for the noise generation
        if len(GlobalFields) != 0 and mode is DataModes.TRAINING and cfg.GLOBALS_NOISE is not None :

            u = np.array([GlobalFields(h) for h in self.halos])
            
            # ok, this is probably not efficient but who cares, these arrays are small
            # and we do this O(1) times
            for ii in range(len(self.halos)) :
                
                # some distributions are two-sided, so let's take two values here
                dglobals = np.empty((len(GlobalFields), 2))

                for jj in range(len(GlobalFields)) :

                    diffs = np.delete(u[:,jj], ii) - u[ii,jj]
                    neg_diffs = - diffs[diffs<0]
                    pos_diffs = diffs[diffs>0]
                    
                    dglobals[jj, 0] = np.min(neg_diffs) if len(neg_diffs)>0 else np.min(pos_diffs)
                    dglobals[jj, 1] = np.min(pos_diffs) if len(pos_diffs)>0 else np.min(neg_diffs)

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
