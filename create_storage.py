import numpy as np

import h5py

from halo import Halo
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

for ii in range(len(halo_catalog['idx_DM'])) :
    
    print(ii)

    h = Halo(halo_catalog, ii)

    s = h.prt_start_DM
    l = h.prt_len_DM

    with h5py.File(cfg.SIM_FILES['DM'], 'r') as f :
        particles = f['Snapshots/%d/PartType%d'%(cfg.SNAP_IDX, cfg.PART_TYPES['DM'])]
        coords = particles['Coordinates'][s:s+l].astype(np.float32)

    np.savez(h.storage_DM, coords=coords)

    s = h.prt_start_TNG
    l = h.prt_len_TNG

    with h5py.File(cfg.SIM_FILES['TNG'], 'r') as f :
        particles = f['Snapshots/%d/PartType%d'%(cfg.SNAP_IDX, cfg.PART_TYPES['TNG'])]
        coords = particles['Coordinates'][s:s+l].astype(np.float32)
        e = particles['InternalEnergy'][s:s+l]
        x = particles['ElectronAbundance'][s:s+l]
        d = particles['Density'][s:s+l]

    XH = 0.76
    gamma = 5.0/3.0
    Pth = 2.0 * (1+XH) / (1 + 3*XH + 4*XH*x) * (gamma - 1) * d * e

    np.savez(h.storage_TNG, coords=coords, Pth=Pth.astype(np.float32))
