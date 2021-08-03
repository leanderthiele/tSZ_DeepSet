"""
A simple script to figure out how the number of DM particles
around randomly chosen positions is distributed.

We require this knowledge to properly standardize the input
for the local network.
"""

import ctypes as ct

import numpy as np

from data_modes import DataModes
from halo_catalog import HaloCatalog
from halo import Halo
import prtfinder
import cfg

N_per_halo = 128

data = []

halos = HaloCatalog(DataModes.TRAINING, compute_dglobals=False)

rng = np.random.default_rng(133)


for halo_idx, halo in enumerate(halos) :

    print('halo_idx : %d'%halo_idx)

    assert isinstance(halo, Halo)

    x = np.fromfile(halo.storage_DM['coords'], dtype=np.float32)
    x = x.reshape((len(x)//3, 3))
    x -= halo.pos
    x[x >  0.5*cfg._BOX_SIZE] -= cfg._BOX_SIZE
    x[x < -0.5*cfg._BOX_SIZE] += cfg._BOX_SIZE

    x_TNG = np.fromfile(halo.storage_TNG['coords'], dtype=np.float32)
    x_TNG = x_TNG.reshape((len(x_TNG)//3, 3))
    x_TNG -= halo.pos
    x_TNG[x_TNG >  0.5*cfg._BOX_SIZE] -= cfg._BOX_SIZE
    x_TNG[x_TNG < -0.5*cfg._BOX_SIZE] += cfg._BOX_SIZE

    if cfg.RMAX is not None :
        r_TNG = np.linalg.norm(x_TNG, axis=-1)
        x_TNG = x_TNG[r_TNG < cfg.RMAX * halo.R200c]

    offsets = np.fromfile(halo.storage_DM['offsets'], dtype=np.uint)

    for ii in range(N_per_halo) :

        x0 = x_TNG[rng.integers(len(x_TNG))]

        if False :
            # turning this segment on enables check against brute-force solution
            # and demonstrates that it really is a good idea to use the more sophisticated
            # compiled version below
            r = np.linalg.norm(x - x0[None,:], axis=-1)
            Nout_reference = np.count_nonzero(r < cfg.R_LOCAL)
        else :
            Nout_reference = None
        
        err = ct.c_int(0)
        Nout = ct.c_uint64(0)
        ul_corner = (np.zeros(3) - 2.51 * halo.R200c).astype(np.float32)
        extent = 2 * 2.51 * halo.R200c

        ptr = prtfinder.prtfinder(x0, cfg.R_LOCAL,
                                  x, len(x),
                                  ul_corner, extent, offsets,
                                  ct.byref(Nout), ct.byref(err))

        err = err.value
        Nout = Nout.value

        if err != 0 :
            raise RuntimeError('prtfinder returned with err=%d'%err)

        data.append(Nout)

        if Nout_reference is not None and Nout != Nout_reference :
            print('Nout=%d -- Nout_reference=%d'%(Nout, Nout_reference))

        prtfinder.myfree(ptr)


np.save('Nlocal_statistics.npy', np.array(data))
