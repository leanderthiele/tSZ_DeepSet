import numpy as np

from data_modes import DataModes
from halo import Halo
from global_fields import GlobalFields
import cfg

class HaloCatalog(list) :

    def __init__(self, mode, compute_dglobals=True) :
        """
        mode ... the mode for this halo catalog
        compute_dglobals ... whether dglobals should be added to the halos for noise generation
        """
    #{{{
        assert isinstance(mode, DataModes)

        halo_catalog = dict(np.load(cfg.HALO_CATALOG))
        Ntot = len(halo_catalog['idx_DM'])

        halos = [Halo(halo_catalog, ii) for ii in mode.sample_indices(Ntot)]

        # figure out the zero mean, unit variance normalization
        # we expect at this point that the halos do not have these fields
        # (although in practice it wouldn't be a problem, it would still be unexpected)
        assert all(not hasattr(h, 'u_avg') and not hasattr(h, 'u_std') for h in halos)
        train_halos = halos if mode is DataModes.TRAINING \
                      else HaloCatalog(DataModes.TRAINING, compute_dglobals=False)
        u = np.array([GlobalFields(h) for h in train_halos])
        u_avg = np.mean(u, axis=0)
        u_std = np.std(u, axis=0)

        # add these (small) arrays to the halos as member variables
        for h in halos :
            h.u_avg = u_avg
            h.u_std = u_std
        
        # if we are training and there are global fields, need to populate the dglobals
        # member variable for the noise generation
        if len(GlobalFields) != 0 and mode is DataModes.TRAINING and cfg.GLOBALS_NOISE is not None :
            
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
