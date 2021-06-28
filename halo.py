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
            if isinstance(v, np.ndarray) :
                assert len(v) == halo_catalog['Nobjects']
                setattr(self, k, v[halo_index])
            else :
                assert k == 'Nobjects'

        self.P200c = self.__P200c(self.M200c, self.R200c)
        self.V200c = self.__V200c(self.M200c, self.R200c)

        self.storage_DM = dict(coords=cfg.STORAGE_FILES['DM']%(halo_index, 'coords'),
                               velocities=cfg.STORAGE_FILES['DM']%(halo_index, 'velocities'))
        self.storage_TNG = dict(# these are the direct output of rockstar_halos/collect_particles.cpp
                                coords=cfg.STORAGE_FILES['TNG']%(halo_index, 'coords'),
                                masses=cfg.STORAGE_FILES['TNG']%(halo_index, 'masses'),
                                Pth=cfg.STORAGE_FILES['TNG']%(halo_index, 'Pth'),
                                # the following store the same arrays with outliers removed
                                coords_filtered=cfg.STORAGE_FILES['TNG']%(halo_index, 'coords_filtered'),
                                masses_filtered=cfg.STORAGE_FILES['TNG']%(halo_index, 'masses_filtered'),
                                Pth_filtered=cfg.STORAGE_FILES['TNG']%(halo_index, 'Pth_filtered'))
    #}}}


    @staticmethod
    def __P200c(M200c, R200c) :
        """
        computes the (thermal!) P200c in Illustris code units
        TODO so far, this is only at z=0
        """
    #{{{
        return 100 * cfg.G_NEWTON  * cfg.RHO_CRIT * cfg.OMEGA_B * M200c / cfg.OMEGA_M / R200c
    #}}}


    @staticmethod
    def __V200c(M200c, R200c) :
        """
        computes the characteristic velocity in Illustris code units
        """
    #{{{
        return np.sqrt(cfg.G_NEWTON * M200c / R200c)
    #}}}
