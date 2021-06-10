import numpy as np
import h5py

import cfg


class DataItem :
    """
    represents one data item, i.e. a collection of network input and target
    """

    def __init__(self, halo, indices=None) :
        """
        halo    ... a Halo instance for the current halo
        indices ... the relative indices, should either be None or have keys
                    'DM', 'TNG'
        
        (we do not want to open the halo catalog file for each construction,
         also maybe the caller wants to modify it in some way)
        """
    #{{{
        self.halo = halo
        self.indices = indices

        self.DM_in = self.__get_DM()
        self.TNG_coords, self.TNG_pressure = self.__get_TNG()
    #}}}

    def __get_DM(self) :
        """
        returns the properties for the network input 
        """
    #{{{
        select = self.__select('DM')

        # FIXME parallel hdf5
        # load the particle coordinates from file
        with h5py.File(cfg.SIM_FILES['DM'], 'r') as f :
            BoxSize = f['Parameters'].attrs['BoxSize']
            particles = f['Snapshots/%d/PartType%d'%(cfg.SNAP_IDX, cfg.PART_TYPES['DM'])]
            coords = self.__read_prt_field(particles, 'Coordinates', 'DM')

        # remove the center of mass
        coords -= self.halo.CM_DM

        # take periodic boundary conditions into account
        coords = DataItem.__periodicize(coords, BoxSize)

        # if required, divide by R200c
        if cfg.NORMALIZE_COORDS :
            coords /= self.halo.R200c_DM

        return coords
    #}}}


    def __get_TNG(self) :
        """
        returns the properties for the network output
        """
    #{{{
        select = self.__select('TNG')

        # FIXME parallel hdf5
        with h5py.File(cfg.SIM_FILES['TNG'], 'r') as f :
            BoxSize = f['Parameters'].attrs['BoxSize']
            particles = f['Snapshots/%d/PartType%d'%(cfg.SNAP_IDX, cfg.PART_TYPES['TNG'])]
            coords = self.__read_prt_field(particles, 'Coordinates', 'TNG')
            e = self.__read_prt_field(particles, 'InternalEnergy', 'TNG')
            x = self.__read_prt_field(particles, 'ElectronAbundance', 'TNG')
            d = self.__read_prt_field(particles, 'Density', 'TNG')

        # compute the electron pressure
        XH = 0.76
        gamma = 5.0/3.0
        pressure = 2.0 * (1+XH) / (1 + 3*XH + 4*XH*x) * (gamma - 1) * d * e

        # remove the center of mass
        coords -= self.halo.CM_DM

        # take periodic boundary conditions into account
        coords = DataItem.__periodicize(coords, BoxSize)

        # if required, divide by R200c
        if cfg.NORMALIZE_COORDS :
            coords /= self.halo.R200c_DM

        return coords, pressure
    #}}}


    @staticmethod
    def __periodicize(x, b) :
        """
        remaps the array x into the interval [0, b] with periodic boundary conditions
        """
    #{{{
        x[x >  0.5*b] -= b
        x[x < -0.5*b] += b

        return x
    #}}}

    
    def __select(self, ptype) :
        """
        returns an object that can be used to index arrays,
        ptype is either 'DM' or 'TNG'
        """
    #{{{
        assert ptype in ['DM', 'TNG']
        indices = None if (self.indices is None or ptype not in self.indices) else self.indices[ptype]
        return indices if indices is not None else slice(None)
    #}}}


    def __read_prt_field(self, group, name, ptype) :
        """
        returns a numpy array of the field <name> in the <group> for the given <ptype>
        """
    #{{{
        assert ptype in ['DM', 'TNG']
        prt_start = self.halo.prt_start_DM if ptype=='DM' else self.halo.prt_start_TNG
        prt_len = self.halo.prt_len_DM if ptype=='DM' else self.halo.prt_len_TNG

        return group[name][prt_start : prt_start+prt_len, :][self.__select(ptype)].astype(np.float32)
    #}}}
