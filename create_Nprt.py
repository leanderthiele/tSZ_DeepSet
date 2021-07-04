import numpy as np

from halo import Halo
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

if 'Nprt_DM' in halo_catalog :
    print('Found Nprt_DM, continuing')
    del halo_catalog['Nprt_DM']
if 'Nprt_TNG' in halo_catalog :
    print('Found Nprt_TNG, continuing')
    del halo_catalog['Nprt_TNG']

Nhalos = halo_catalog['Nobjects']

Nprt_DM = np.empty(Nhalos, dtype=int)
Nprt_TNG = np.empty(Nhalos, dtype=int)

for ii in range(Nhalos) :
    
    h = Halo(halo_catalog, ii)

    xDM = np.fromfile(h.storage_DM['coords'], dtype=np.float32)
    Nprt_DM[ii] = len(xDM) / 3
    vDM = np.fromfile(h.storage_DM['velocities'], dtype=np.float32)
    assert 3 * Nprt_DM[ii] == len(vDM)

    xTNG = np.fromfile(h.storage_TNG['coords'], dtype=np.float32)
    Nprt_TNG[ii] = len(xTNG) / 3
    Pth = np.fromfile(h.storage_TNG['Pth'], dtype=np.float32)
    assert Nprt_TNG[ii] == len(Pth)

np.savez(cfg.HALO_CATALOG, **halo_catalog, Nprt_DM=Nprt_DM, Nprt_TNG=Nprt_TNG)
