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

rng = np.random.default_rng()


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

    offsets = np.fromfile(halo.storage_DM['offsets'], dtype=np.int64)

    for ii in range(N_per_halo) :

        print('\tii : %d'%ii)

        x0 = x_TNG[rng.integers(len(x_TNG))]
        
        err = ct.c_int(0)
        Nout = ct.c_uint64(0)
        ul_corner = -2.51 * halo.R200c
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

        r = np.linalg.norm(x - x0, axis=-1)
        Nout_reference = np.count_nonzero(r < R)
        print('Nout=%d -- Nout_reference=%d'%(Nout, Nout_reference))

        prtfinder.myfree(ptr)


np.save('Nlocal_statistics.npy', np.array(data))
