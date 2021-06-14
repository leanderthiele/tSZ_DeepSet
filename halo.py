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
        self.P200c_DM = self.__P200c(self.M200c_DM, self.R200c_DM)
        self.prt_start_DM = get_entry('prt_start_DM')
        self.prt_len_DM = get_entry('prt_len_DM')

        self.idx_TNG = get_entry('idx_TNG')
        self.pos_TNG = get_entry('pos_TNG')
        self.CM_TNG = get_entry('CM_TNG')
        self.M200c_TNG = get_entry('M200c_TNG')
        self.R200c_TNG = get_entry('R200c_TNG')
        self.P200c_TNG = self.__P200c(self.M200c_TNG, self.R200c_TNG)
        self.prt_start_TNG = get_entry('prt_start_TNG')
        self.prt_len_TNG = get_entry('prt_len_TNG')

        self.storage_DM = cfg.STORAGE_FILES['DM']%self.idx_DM
        self.storage_TNG = cfg.STORAGE_FILES['TNG']%self.idx_TNG
    #}}}


    def __P200c(self, M200c, R200c) :
        """
        computes the (thermal!) P200c in Illustris code units
        TODO so far, this is only at z=0
        """
    #{{{
        rho_crit = 2.775e-8 # critical density at z=0 in Illustris code units
        G = 4.30091e4 # Newton's constant in Illustris code units
        Ob = 0.0486
        Om = 0.3089

        return 100 * G  * rho_crit * Ob * M200c / Om / R200c
    #}}}
