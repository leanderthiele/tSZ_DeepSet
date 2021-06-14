import numpy as np
import h5py

from origin import Origin
import cfg


class DataItem :
    """
    represents one data item, i.e. a collection of network input and target

    Upon construction, the instance has the fields:
        halo         ... the Halo instance this data item is associated with
        DM_coords    ... the coordinates of the dark matter particles
        TNG_coords   ... the coordinates of the gas particles
        TNG_Pth      ... the electron pressure at the position of those gas particles
        TNG_radii    ... the radial coordinates of the gas particles (with the last dimension length 1)
    """

    def __init__(self, halo, indices=None,
                       load_DM=True, load_TNG=True, origin=Origin.CM, compute_TNG_radii=False) :
        """
        halo     ... a Halo instance for the current halo
        indices  ... the relative indices, should either be None or have keys
                     'DM', 'TNG'
        load_DM  ... whether to load the dark matter particles
        load_TNG ... whether to load the TNG particles
        origin   ... definition of the origin of our coordinate system, either CM or POS
        compute_TNG_radii ... whether the radial positions of the TNG particles are to be computed
        
        (we do not want to open the halo catalog file for each construction,
         also maybe the caller wants to modify it in some way)
        """
    #{{{
        self.halo = halo
        self.indices = indices

        self.has_DM = load_DM
        self.has_TNG = load_TNG
        self.has_TNG_radii = compute_TNG_radii

        assert origin in [Origin.CM, Origin.POS, ]
        self.origin = origin

        if load_DM :
            self.DM_coords = self.__get_DM()

        if load_TNG :
            self.TNG_coords, self.TNG_Pth = self.__get_TNG()

            if compute_TNG_radii :
                # compute the scalar distances if coordinates are already in the correct frame
                self.TNG_radii = np.linalg.norm(self.TNG_coords, axis=-1, keepdims=True)
    #}}}


    def __get_DM(self) :
        """
        returns the properties for the network input 
        """
    #{{{
        select = self.__select('DM')

        # load the particle coordinates from file
        with np.load(self.halo.storage_DM) as f :
            coords = f['coords'][select]

        # remove the origin if required
        if self.origin is Origin.CM :
            coords -= self.halo.CM_DM
        elif self.origin is Origin.POS :
            coords -= self.halo.pos_DM

        # take periodic boundary conditions into account
        coords = DataItem.__periodicize(coords)

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

        # load particle coordinates and thermal pressure at their position
        # from file
        with np.load(self.halo.storage_TNG) as f :
            coords = f['coords'][select]
            Pth = f['Pth'][select]

        # remove the origin if required 
        if self.origin is Origin.CM :
            coords -= self.halo.CM_DM
        elif self.origin is Origin.POS :
            coords -= self.halo.pos_DM

        # take periodic boundary conditions into account
        coords = DataItem.__periodicize(coords)

        # if required, divide by R200c
        if cfg.NORMALIZE_COORDS :
            coords /= self.halo.R200c_DM

        # normalize the thermal pressure
        Pth /= self.halo.P200c_DM

        return coords, Pth[:, None]
    #}}}


    @staticmethod
    def __periodicize(x) :
        """
        remaps the array x into the interval [-BoxSize/2, +BoxSize/2] with periodic boundary conditions
        """
    #{{{
        x[x >  0.5*cfg.BOX_SIZE] -= cfg.BOX_SIZE
        x[x < -0.5*cfg.BOX_SIZE] += cfg.BOX_SIZE

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
