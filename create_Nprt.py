import numpy as np

from halo import Halo
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

res_str = '%d_'%cfg.TNG_RESOLUTION if cfg.TNG_RESOLUTION != 256 else ''

Nprt_DM_str = 'Nprt_%sDM'%res_str
Nprt_TNG_str = 'Nprt_%sTNG'%res_str

if Nprt_DM_str in halo_catalog :
    print('Found %s, continuing'%Nprt_DM_str)
    del halo_catalog[Nprt_DM_str]
if Nprt_TNG_str in halo_catalog :
    print('Found %s, continuing'%Nprt_TNG_str)
    del halo_catalog[Nprt_TNG_str]

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

np.savez(cfg.HALO_CATALOG, **halo_catalog,
         **{Nprt_DM_str: Nprt_DM, Nprt_TNG_str: Nprt_TNG})
