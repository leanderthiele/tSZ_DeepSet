import numpy as np

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

    def __init__(self, halo, mode,
                       load_DM=True, load_TNG=True,
                       origin=Origin.CM, compute_TNG_radii=False) :
        """
        halo     ... a Halo instance for the current halo
        mode     ... the mode this item was loaded in
        load_DM  ... whether to load the dark matter particles
        load_TNG ... whether to load the TNG particles
        origin   ... definition of the origin of our coordinate system, either CM or POS
        compute_TNG_radii ... whether the radial positions of the TNG particles are to be computed
        
        (we do not want to open the halo catalog file for each construction,
         also maybe the caller wants to modify it in some way)
        """
    #{{{
        self.halo = halo
        self.mode = mode

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
        # load the particle coordinates from file
        with np.load(self.halo.storage_DM) as f :
            coords = f['coords']

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
        # load particle coordinates and thermal pressure at their position
        # from file
        with np.load(self.halo.storage_TNG) as f :
            coords = f['coords']
            Pth = f['Pth']

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


    def sample_particles(self, indices) :
        """
        returns a copy of this data item with only a subset of the particles randomly sampled.
        (this instance is not modified!)
        indices ... a dict with keys 'DM', 'TNG', each an array of integers corresponding to the particle
                    indices
        """
    #{{{
        # construct a new DataItem without any data
        out = DataItem(self.halo, self.mode, load_DM=False, load_TNG=False)

        # copy our fields
        out.origin = self.origin
        out.has_DM = self.has_DM
        out.has_TNG = self.has_TNG
        out.has_TNG_radii = self.has_TNG_radii

        # now fill in the sampled particles

        if self.has_DM :
            out.DM_coords = self.DM_coords[indices['DM']]

        if self.has_TNG :
            out.TNG_coords = self.TNG_coords[indices['TNG']]
            out.TNG_Pth = self.TNG_Pth[indices['TNG']]

            if self.has_TNG_radii :
                out.TNG_radii = self.TNG_radii[indices['TNG']]

        # give this instance some unique hash depending on the random indices passed
        # NOTE this hash is not necessarily positive, as the sums may well overflow
        out.hash = ( np.sum(indices['DM']) if self.has_DM else 0 ) \
                   + ( np.sum(indices['TNG']) if self.has_TNG else 0 ) \
                   + self.halo.idx_DM

        return out
    #}}}
