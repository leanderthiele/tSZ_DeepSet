import numpy as np

import cfg


class Halo :
    """
    A wrapper around a single entry in the halo catalog
    """

    def __init__(self, halo_catalog, halo_index) :
        """
        we avoid reading the halo catalog file repeatedly by letting the caller
        load it into memory as a dict and pass it to this constructor
        """
    #{{{ 
        # copy all the entries in the halo catalog that make sense
        # into this instance

        for k, v in halo_catalog.items() :
            if k == 'Nobjects' :
                continue
            assert len(v) == halo_catalog['Nobjects']
            setattr(self, k, v[halo_index])

        self.P200c = Halo.__P200c(self.M200c, self.R200c)
        self.V200c = Halo.__V200c(self.M200c, self.R200c)

        self.storage_DM = dict(coords=cfg._STORAGE_FILES['DM']%(halo_index, 'coords'),
                               velocities=cfg._STORAGE_FILES['DM']%(halo_index, 'velocities'),
                               offsets=cfg._STORAGE_FILES['DM']%(halo_index, 'offsets'))
        
        self.storage_TNG = dict(coords=cfg._STORAGE_FILES['TNG']%(halo_index, 'box_%d_coords'%cfg.TNG_RESOLUTION),
                                Pth=cfg._STORAGE_FILES['TNG']%(halo_index, 'box_%d_Pth'%cfg.TNG_RESOLUTION),
                                residuals=cfg._STORAGE_FILES['TNG']%(halo_index, 'residuals'))

        self.idx = halo_index
    #}}}


    @staticmethod
    def __P200c(M200c, R200c) :
        """
        computes the (thermal!) P200c in Illustris code units
        TODO so far, this is only at z=0
        """
    #{{{
        return 100 * cfg._G_NEWTON  * cfg._RHO_CRIT * cfg._OMEGA_B * M200c / cfg._OMEGA_M / R200c
    #}}}


    @staticmethod
    def __V200c(M200c, R200c) :
        """
        computes the characteristic velocity in Illustris code units
        """
    #{{{
        return np.sqrt(cfg._G_NEWTON * M200c / R200c)
    #}}}
