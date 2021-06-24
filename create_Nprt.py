import numpy as np

from halo import Halo
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

Nhalos = len(halo_catalog['idx_DM'])

Nprt_DM = np.empty(Nhalos, dtype=int)
Nprt_TNG = np.empty(Nhalos, dtype=int)

for ii in range(Nhalos) :
    
    h = Halo(halo_catalog, ii)

    with np.load(h.storage_DM) as f :
        Nprt_DM[ii] = len(f['coords'])

    with np.load(h.storage_TNG) as f :
        Nprt_TNG[ii] = len(f['coords'])

np.savez(cfg.HALO_CATALOG, **halo_catalog, Nprt_DM=Nprt_DM, Nprt_TNG=Nprt_TNG)
