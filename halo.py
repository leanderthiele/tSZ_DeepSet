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

        self.M200c = get_entry('M200c')
        self.R200c = get_entry('R200c')
        self.P200c = self.__P200c(self.M200c, self.R200c)
        self.V200c = self.__V200c(self.M200c, self.R200c)

        self.pos = get_entry('pos')
        self.CM = get_entry('CM') # note : this is a vector with respect to the pos!
        self.min_pot_pos_DM = get_entry('min_pot_pos_DM')
        self.min_pot_pos_TNG = get_entry('min_pot_pos_TNG')

        self.vel = get_entry('vel')
        self.vel_dispersion = get_entry('vel_dispersion')
        self.inertia = get_entry('inertia')
        self.ang_momentum = get_entry('ang_mom')
        self.ang_momentum2 = get_entry('ang_mom2')

        self.Xoff = get_entry('Xoff')
        self.Voff = get_entry('Voff')
        self.Vmax = get_entry('Vmax')
        self.Vrms = get_entry('Vrms')
        self.Rs = get_entry('Rs')
        self.rs_klypin = get_entry('rs_klypin')
        self.M200c_all = get_entry('M200c_all')
        self.Mvir = get_entry('Mvir')
        self.M200b = get_entry('M200b')
        self.M500c = get_entry('M500c')
        self.M2500c = get_entry('M2500c')
        self.Spin = get_entry('Spin')
        self.spin_bullock = get_entry('spin_bullock')
        self.b_to_a = get_entry('b_to_a')
        self.c_to_a = get_entry('c_to_a')

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

        if 'Nprt_DM' in halo_catalog :
            self.Nprt_DM = get_entry('Nprt_DM')
        if 'Nprt_TNG' in halo_catalog :
            self.Nprt_TNG = get_entry('Nprt_TNG')
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
