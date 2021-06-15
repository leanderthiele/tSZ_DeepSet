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
        get_entry = lambda name : halo_catalog[name][halo_index]
        
        self.idx_DM = get_entry('idx_DM')
        self.pos_DM = get_entry('pos_DM')
        self.CM_DM = get_entry('CM_DM')
        self.M200c_DM = get_entry('M200c_DM')
        self.R200c_DM = get_entry('R200c_DM')
        self.P200c_DM = Halo.__P200c(self.M200c_DM, self.R200c_DM)
        self.V200c_DM = Halo.__V200c(self.M200c_DM, self.R200c_DM)
        self.prt_start_DM = get_entry('prt_start_DM')
        self.prt_len_DM = get_entry('prt_len_DM')

        self.inertia_DM = get_entry('inertia_DM')
        self.ang_momentum_DM = get_entry('ang_momentum_DM')
        self.central_CM = get_entry('central_CM')

        self.idx_TNG = get_entry('idx_TNG')
        self.pos_TNG = get_entry('pos_TNG')
        self.CM_TNG = get_entry('CM_TNG')
        self.M200c_TNG = get_entry('M200c_TNG')
        self.R200c_TNG = get_entry('R200c_TNG')
        self.P200c_TNG = Halo.__P200c(self.M200c_TNG, self.R200c_TNG)
        self.V200c_TNG = Halo.__V200c(self.M200c_TNG, self.R200c_TNG)
        self.prt_start_TNG = get_entry('prt_start_TNG')
        self.prt_len_TNG = get_entry('prt_len_TNG')

        self.storage_DM = cfg.STORAGE_FILES['DM']%self.idx_DM
        self.storage_TNG = cfg.STORAGE_FILES['TNG']%self.idx_TNG
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
