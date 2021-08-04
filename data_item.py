import ctypes as ct

import numpy as np

import prtfinder
import cfg


class DataItem :
    """
    represents one data item, i.e. a collection of network input and target

    Upon construction, the instance has the fields:
        halo          ... the Halo instance this data item is associated with
        DM_coords     ... the coordinates of the dark matter particles
        TNG_coords    ... the coordinates of the gas particles
        TNG_Pth       ... the electron pressure at the position of those gas particles
        TNG_radii     ... the radial coordinates of the gas particles (with the last dimension length 1)
        TNG_residuals ... the Pth residuals with respect to a simple model, binned and normalized
    """



    def __init__(self, halo, mode,
                       load_DM=True, load_TNG=True, load_TNG_residuals=True) :
        """
        halo     ... a Halo instance for the current halo
        mode     ... the mode this item was loaded in
        load_DM  ... whether to load the dark matter particles
        load_TNG ... whether to load the TNG particles
        load_TNG_residuals ... whether the TNG residuals are to be loaded (these files are small
                               so we load them regarless of load_TNG)
        
        (we do not want to open the halo catalog file for each construction,
         also maybe the caller wants to modify it in some way)
        """
    #{{{
        self.halo = halo
        self.mode = mode

        self.has_DM = load_DM
        self.has_TNG = load_TNG
        self.has_TNG_residuals = load_TNG_residuals

        if load_DM :
            self.DM_coords, self.DM_vels, self.__DM_offsets = self.__get_DM()
            assert (self.DM_vels is None and not cfg.USE_VELOCITIES) \
                   or (self.DM_vels is not None and cfg.USE_VELOCITIES)

        if load_TNG :
            self.TNG_coords, self.TNG_Pth = self.__get_TNG()
            self.TNG_radii = np.linalg.norm(self.TNG_coords, axis=-1, keepdims=True)
            if cfg.RMAX is not None :
                mask = self.TNG_radii < (cfg.RMAX if cfg.NORMALIZE_COORDS else cfg.RMAX * self.halo.R200c)
                self.TNG_coords = self.TNG_coords[mask]
                self.TNG_Pth = self.TNG_Pth[mask]
                self.TNG_radii = self.TNG_radii[mask]
                del mask

        if load_TNG_residuals :
            self.TNG_residuals = self.__get_TNG_residuals()
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

        if cfg.NET_ARCH['local'] :
            offsets = np.fromfile(self.halo.storage_DM['offsets'], dtype=np.uint)
        else :
            offsets = None

        return coords, vels, offsets
    #}}}


    def __get_TNG(self) :
        """
        returns the properties for the network output
        """
    #{{{
        # load particle coordinates and thermal pressure at their position from file
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


    def __get_TNG_residuals(self) :
    #{{{
        return np.fromfile(self.halo.storage_TNG['residuals'], dtype=np.float32)
    #}}}


    def __get_DM_local_indices(self, x_TNG) :
        """
        returns indices of all DM particles within a certain distance of x_TNG
        (both as raw pointer and numpy array)
        """
    #{{{
        R = cfg.R_LOCAL / (self.halo.R200c if cfg.NORMALIZE_COORDS else 1)
        
        # passed by reference
        err = ct.c_int(0) # error status
        Nout = ct.c_uint64(0) # length of the returned array

        # geometry
        ul_corner = np.full(3,
                            -2.51 * (1 if cfg.NORMALIZE_COORDS else self.halo.R200c),
                            dtype=np.float32)
        extent = 2 * 2.51 * (1 if cfg.NORMALIZE_COORDS else self.halo.R200c)

        # call the compiled library
        ptr = prtfinder.prtfinder(x_TNG, R,
                                  self.DM_coords, len(self.DM_coords),
                                  ul_corner, extent, self.__DM_offsets,
                                  ct.byref(Nout), ct.byref(err))

        # convert to native python
        err = err.value
        Nout = Nout.value

        # check if error occured
        if err != 0 :
            raise RuntimeError('prtfinder returned with err=%d'%err)

        if Nout == 0 :
            # pathological case: we have no DM particles in the vicinity
            # (for reasonable choices of cfg.R_LOCAL, this is a permille event)
            return ptr, None

        # convert raw pointer into numpy array
        return ptr, np.ctypeslib.as_array(ptr, shape=(Nout,))
    #}}}


    def __get_DM_local(self, TNG_coords, rng) :
        """
        get local dark matter around the given TNG coordinates
        returns Nparticles, coordinates, velocities
        """
    #{{{
        assert isinstance(rng, np.random.generator.Generator)

        # initialize these arrays such that the initial values make sense for the case
        # when no DM particles are in the vicinity
        # (the choice of N=1 is good because it prevents anything from blowing up)
        N = np.ones(len(TNG_coords), dtype=np.float32)
        x = np.zeros((len(TNG_coords), int(cfg.N_LOCAL), 3), dtype=np.float32)
        v = np.zeros((len(TNG_coords), int(cfg.N_LOCAL), 3), dtype=np.float32)


        for ii, x_TNG in enumerate(TNG_coords) :

            raw_ptr, prt_indices = self.__get_DM_local_indices(x_TNG)

            if prt_indices is not None :
                # there are DM particles in the vicinity (this is the 99+% case),
                # so we sample some of them

                N[ii] = len(prt_indices)

                prt_indices = prt_indices[rng.integers(N[ii], size=int(cfg.N_LOCAL))]

                x[ii, ...] = self.DM_coords[prt_indices]
                v[ii, ...] = self.DM_vels[prt_indices]

            # now we can safely free the memory
            prtfinder.myfree(raw_ptr)

        
        return N, x, v
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


    def sample_particles(self, indices, TNG_residuals_noise_rng=None, local_rng=None) :
        """
        returns a copy of this data item with only a subset of the particles randomly sampled.
        (this instance is not modified!)
        indices ... a dict with keys 'DM', 'TNG', each an array of integers corresponding to the particle
                    indices
        Also gives the option to add noise to the TNG residuals, by passing a numpy rng.
        """
    #{{{
        # construct a new DataItem without any data
        out = DataItem(self.halo, self.mode, load_DM=False, load_TNG=False, load_TNG_residuals=False)

        # copy our fields
        out.has_DM = self.has_DM
        out.has_TNG = self.has_TNG
        out.has_TNG_residuals = self.has_TNG_residuals

        # clamp the indices to the allowed range
        indices['DM'] %= len(self.DM_coords)
        indices['TNG'] %= len(self.TNG_coords)

        # now fill in the sampled particles

        if self.has_DM :
            out.DM_coords = self.DM_coords[indices['DM']]
            if self.DM_vels is not None :
                out.DM_vels = self.DM_vels[indices['DM']]

        if self.has_TNG :
            out.TNG_coords = self.TNG_coords[indices['TNG']]
            out.TNG_Pth = self.TNG_Pth[indices['TNG']]
            out.TNG_radii = self.TNG_radii[indices['TNG']]

        if self.has_TNG_residuals :
            out.TNG_residuals = self.TNG_residuals
            if TNG_residuals_noise_rng is not None and cfg.RESIDUALS_NOISE is not None :
                out.TNG_residuals += TNG_residuals_noise_rng.normal(scale=cfg.RESIDUALS_NOISE,
                                                                    size=cfg.RESIDUALS_NBINS)

        if self.has_DM and self.has_TNG and cfg.NET_ARCH['local'] :
            out.DM_N_local, out.DM_coords_local, out.DM_vels_local \
                = self.__get_DM_local(out.TNG_coords, local_rng)

        # give this instance some unique hash depending on the random indices passed
        # NOTE this hash is not necessarily positive, as the sums may well overflow
        out.hash = ( np.sum(indices['DM']) if self.has_DM else 0 ) \
                   + ( np.sum(indices['TNG']) if self.has_TNG else 0 ) \
                   + hash(self.halo.M200c)

        return out
    #}}}
