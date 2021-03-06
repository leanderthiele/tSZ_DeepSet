import os
import ctypes as ct

import numpy as np

import prtfinder
import cfg


class DataItem :
    """
    represents one data item, i.e. a collection of network input and target

    Upon construction, the instance has the fields:
        [reference]
        halo          ... the Halo instance this data item is associated with
        mode          ... the mode argument from the init call

        [bool]
        has_DM
        has_TNG
        has_TNG_residuals

        [data]
        DM_coords     ... the coordinates of the dark matter particles
        DM_vels       ... the velocities of the dark matter particles
        TNG_coords    ... the coordinates of the gas particles
        TNG_Pth       ... the electron pressure at the position of those gas particles
        TNG_radii     ... the radial coordinates of the gas particles (with the last dimension length 1)
        TNG_residuals ... the Pth residuals with respect to a simple model, binned and normalized
    These fields are NOT ready for use yet, usable DataItem instances are always
    constructed using the sample_particles() method.

    Calling sample_particles() returns a DataItem instance with usable fields:
        [reference]
        halo
        mode

        [bool]
        has_DM
        has_TNG
        has_TNG_residuals
        has_DM_local

        [data]
        DM_coords
        DM_vels
        TNG_coords
        TNG_Pth
        TNG_radii
        TNG_residuals
        DM_N_local
        DM_coords_local
        DM_vels_local
    In particular, except for the `local' data everything is normalized by 200c quantities now!
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
            self.__N_DM, self.DM_coords, self.DM_vels, self.__DM_offsets = self.__get_DM()
            assert (self.DM_vels is None and not cfg.USE_VELOCITIES) \
                   or (self.DM_vels is not None and cfg.USE_VELOCITIES)

        if load_TNG :
            self.TNG_coords, self.TNG_Pth = self.__get_TNG()
            self.TNG_radii = np.linalg.norm(self.TNG_coords, axis=-1, keepdims=True)
            if cfg.RMAX is not None :
                # remember that TNG_radii has a singleton dimension
                mask = self.TNG_radii.flatten() < cfg.RMAX * self.halo.R200c
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
        # first find the number of DM particles
        with open(self.halo.storage_DM['coords'], 'r') as f :
            f.seek(0, os.SEEK_END)
            Nbytes = f.tell()
            N = Nbytes // (3*4) # each particle has 3 32bit numbers
            assert N * 3 * 4 == Nbytes


        if cfg.MEMMAP_DM :
            # do not load the DM particles directly into memory

            coords = np.memmap(self.halo.storage_DM['coords'],
                               mode='r', dtype=np.float32, shape=(N,3,))

            if cfg.USE_VELOCITIES :
                vels = np.memmap(self.halo.storage_DM['velocities'],
                                 mode='r', dtype=np.float32, shape=(N,3,))
            else :
                vels = None

        else :
            # load the DM particles directly into memory (takes longer initially)

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
                assert N == len(coords)
                # subtract bulk motion
                vels -= self.halo.vel
            else :
                vels = None

        if cfg.NET_ARCH['local'] :
            offsets = np.fromfile(self.halo.storage_DM['offsets'], dtype=np.uint)
        else :
            offsets = None

        return N, coords, vels, offsets
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

        return coords, Pth[:, None]
    #}}}


    def __get_TNG_residuals(self) :
    #{{{
        # we have a single NaN in this data, due to an empty radial bin.
        # Should not be problematic.
        return np.nan_to_num(np.fromfile(self.halo.storage_TNG['residuals'], dtype=np.float32))
    #}}}


    def __get_DM_local_indices(self, x_TNG) :
        """
        returns indices of all DM particles within a certain distance of x_TNG
        (both as raw pointer and numpy array)
        """
    #{{{
        # passed by reference
        err = ct.c_int(0) # error status
        Nout = ct.c_uint64(0) # length of the returned array

        # call the compiled library
        ptr = prtfinder.prtfinder(x_TNG, cfg.R_LOCAL, not cfg.MEMMAP_DM,
                                  self.DM_coords if not cfg.MEMMAP_DM else np.empty((1,3), dtype=np.float32),
                                  prtfinder.c_str(self.halo.storage_DM['coords'] if cfg.MEMMAP_DM else 'no file'),
                                  self.__N_DM, cfg._BOX_SIZE, self.halo.R200c, self.halo.pos,
                                  self.__DM_offsets,
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
        # initialize these arrays such that the initial values make sense for the case
        # when no DM particles are in the vicinity
        # (the choice of N=0.9 is good because it prevents anything from blowing up)
        N = np.full(len(TNG_coords), 0.9, dtype=np.float32)

        x = np.zeros((len(TNG_coords), int(cfg.N_LOCAL[str(self.mode)]), 3), dtype=np.float32)

        if self.DM_vels is not None :
            v = np.zeros((len(TNG_coords), int(cfg.N_LOCAL[str(self.mode)]), 3), dtype=np.float32)
        else :
            v = None


        for ii, x_TNG in enumerate(TNG_coords) :

            raw_ptr, prt_indices = self.__get_DM_local_indices(x_TNG)

            if prt_indices is not None :
                # there are DM particles in the vicinity (this is the 99+% case),
                # so we sample some of them

                N[ii] = len(prt_indices)

                prt_indices = prt_indices[rng.integers(N[ii], size=int(cfg.N_LOCAL[str(self.mode)]))]

                x[ii, ...] = self.DM_coords[prt_indices]

                if self.DM_vels is not None :
                    v[ii, ...] = self.DM_vels[prt_indices]
                
                # if we use memmap-ed arrays, these are directly from disk so we need to normalize them
                if cfg.MEMMAP_DM :
                    x[ii, ...] -= self.halo.pos
                    x[ii, ...] = DataItem.__periodicize(x[ii, ...])
                    if self.DM_vels is not None :
                        v[ii, ...] -= self.halo.vel

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
        out.has_DM_local = self.has_DM and self.has_TNG and cfg.NET_ARCH['local']

        # clamp the indices to the allowed range
        indices['DM'] %= len(self.DM_coords)
        indices['TNG'] %= len(self.TNG_coords)

        # now fill in the sampled particles

        if out.has_DM :
            out.DM_coords = self.DM_coords[indices['DM']]
            if self.DM_vels is not None :
                out.DM_vels = self.DM_vels[indices['DM']]
            else :
                out.DM_vels = None
            
            if cfg.MEMMAP_DM :
                # if we use memmap-ed arrays, these are directly from disk so we need to normalize them
                out.DM_coords -= self.halo.pos
                out.DM_coords = DataItem.__periodicize(out.DM_coords)
                if out.DM_vels is not None :
                    out.DM_vels -= self.halo.vel

        if out.has_TNG :
            out.TNG_coords = self.TNG_coords[indices['TNG']]
            out.TNG_Pth = self.TNG_Pth[indices['TNG']]
            out.TNG_radii = self.TNG_radii[indices['TNG']]

        if out.has_TNG_residuals :
            out.TNG_residuals = self.TNG_residuals
            if TNG_residuals_noise_rng is not None and cfg.RESIDUALS_NOISE is not None :
                out.TNG_residuals += TNG_residuals_noise_rng.normal(scale=cfg.RESIDUALS_NOISE,
                                                                    size=cfg.RESIDUALS_NBINS)

        if out.has_DM_local :
            out.DM_N_local, out.DM_coords_local, out.DM_vels_local \
                = self.__get_DM_local(out.TNG_coords, local_rng)

        # normalize the local quantities by some reasonable values to get them to O(1),
        # the standard-normal thing is done later
        # NOTE it is important that this happens before the 200c normalization below
        #      because we need to use the kpc-unit TNG coordinates
        if out.has_DM_local :

            # where we need to shift the DM coordinates
            mask = out.DM_N_local >= 1.0
            
            # take DM coordinates relative to TNG position
            # (but only if the coordinates are real, otherwise they are zero'ed and we shouldn't worry
            out.DM_coords_local[mask] -= out.TNG_coords[mask, None, :]

            # get DM coordinates to O(1)
            out.DM_coords_local /= cfg.R_LOCAL

            if out.DM_vels_local is not None :
                out.DM_vels_local /= 510.0 # km/s, number is meaningless but gives O(1) dispersions

        # normalize the halo-wide quantities by self-similar scales
        if out.has_DM :
            out.DM_coords /= self.halo.R200c
            if out.DM_vels is not None :
                out.DM_vels /= self.halo.V200c

        if out.has_TNG :
            out.TNG_coords /= self.halo.R200c
            out.TNG_radii /= self.halo.R200c
            if cfg.SCALE_PTH :
                out.TNG_Pth /= self.halo.P200c

        # NOTE the TNG residuals are directly from file and already normalized by P200c

        # give this instance some unique hash depending on the random indices passed
        # Will be used to seed the RNG that generates noise on globals and basis
        # NOTE this hash is not necessarily positive, as the sums may well overflow
        out.hash = ( np.sum(indices['DM']) if self.has_DM else 0 ) \
                   + ( np.sum(indices['TNG']) if self.has_TNG else 0 ) \
                   + hash(self.halo.M200c)

        return out
    #}}}
