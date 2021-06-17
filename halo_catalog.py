import numpy as np

from data_modes import DataModes
from halo import Halo
from global_fields import GlobalFields
import cfg

class HaloCatalog(list) :

    def __init__(self, mode) :
        """
        mode ... the mode for this halo catalog
        """
    #{{{
        assert isinstance(mode, DataModes)

        halo_catalog = dict(np.load(cfg.HALO_CATALOG))
        Ntot = len(halo_catalog['idx_DM'])

        halos = [Halo(halo_catalog, ii) for ii in mode.sample_indices(Ntot)]
        
        # if we are training and there are global fields, need to populate the dglobals
        # member variable for the noise generation
        if len(GlobalFields) != 0 and mode is DataModes.TRAINING and cfg.GLOBALS_NOISE is not None :

            u = np.array([GlobalFields(h) for h in halos])
            
            # ok, this is probably not efficient but who cares, these arrays are small
            # and we do this O(1) times
            for ii in range(len(halos)) :
                
                # some distributions are two-sided, so let's take two values here
                dglobals = np.empty((len(GlobalFields), 2))

                for jj in range(len(GlobalFields)) :

                    diffs = np.delete(u[:,jj], ii) - u[ii,jj]
                    neg_diffs = - diffs[diffs<0]
                    pos_diffs = diffs[diffs>0]
                    
                    dglobals[jj, 0] = np.min(neg_diffs) if len(neg_diffs)>0 else np.min(pos_diffs)
                    dglobals[jj, 1] = np.min(pos_diffs) if len(pos_diffs)>0 else np.min(neg_diffs)

                halos[ii].dglobals = dglobals 

        super().__init__(halos)
    #}}}
