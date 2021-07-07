import numpy as np

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
                       compute_TNG_radii=True) :
        """
        halo     ... a Halo instance for the current halo
        mode     ... the mode this item was loaded in
        load_DM  ... whether to load the dark matter particles
        load_TNG ... whether to load the TNG particles
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

        if load_DM :
            self.DM_coords, self.DM_vels = self.__get_DM()
            assert (self.DM_vels is None and not cfg.USE_VELOCITIES) \
                   or (self.DM_vels is not None and cfg.USE_VELOCITIES)

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
        coords = np.fromfile(self.halo.storage_DM['coords'], dtype=np.float32)
        coords = coords.reshape((len(coords)//3, 3))

        # remove the origin
        coords -= self.halo.pos

        # take periodic boundary conditions into account
        coords = DataItem.__periodicize(coords)

        # load the velocities if required
        if cfg.USE_VELOCITIES :
            vels = np.fromfile(self.halo.storage_DM['velocities'], dtype=np.float32)
            vels = vels.reshape((len(vels)//3, 3))
            assert len(vels) == len(coords)
            # subtract bulk motion
            vels -= self.halo.vel
        else :
            vels = None

        # if required, divide by R200c
        if cfg.NORMALIZE_COORDS :
            coords /= self.halo.R200c
            if vels is not None :
                vels /= self.halo.V200c

        return coords, vels
    #}}}


    def __get_TNG(self) :
        """
        returns the properties for the network output
        """
    #{{{
        # load particle coordinates and thermal pressure at their position
        # from file
        coords = np.fromfile(self.halo.storage_TNG['coords'], dtype=np.float32)
        coords = coords.reshape((len(coords)//3, 3))
        Pth = np.fromfile(self.halo.storage_TNG['Pth'], dtype=np.float32)

        # remove the origin
        coords -= self.halo.pos

        # take periodic boundary conditions into account
        coords = DataItem.__periodicize(coords)

        # if required, divide by R200c
        if cfg.NORMALIZE_COORDS :
            coords /= self.halo.R200c

        # normalize the thermal pressure
        Pth /= self.halo.P200c

        return coords, Pth[:, None]
    #}}}


    @staticmethod
    def __periodicize(x) :
        """
        remaps the array x into the interval [-BoxSize/2, +BoxSize/2] with periodic boundary conditions
        """
    #{{{
        x[x >  0.5*cfg._BOX_SIZE] -= cfg._BOX_SIZE
        x[x < -0.5*cfg._BOX_SIZE] += cfg._BOX_SIZE

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
        out.has_DM = self.has_DM
        out.has_TNG = self.has_TNG
        out.has_TNG_radii = self.has_TNG_radii

        # now fill in the sampled particles

        if self.has_DM :
            out.DM_coords = self.DM_coords[indices['DM']]
            if self.DM_vels is not None :
                out.DM_vels = self.DM_vels[indices['DM']]

        if self.has_TNG :
            out.TNG_coords = self.TNG_coords[indices['TNG']]
            out.TNG_Pth = self.TNG_Pth[indices['TNG']]

            if self.has_TNG_radii :
                out.TNG_radii = self.TNG_radii[indices['TNG']]

        # give this instance some unique hash depending on the random indices passed
        # NOTE this hash is not necessarily positive, as the sums may well overflow
        out.hash = ( np.sum(indices['DM']) if self.has_DM else 0 ) \
                   + ( np.sum(indices['TNG']) if self.has_TNG else 0 ) \
                   + hash(self.halo.M200c)

        return out
    #}}}
