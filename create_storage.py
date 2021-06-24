import numpy as np
from scipy.stats import median_abs_deviation

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
        BoxSize = f['Header'].attrs['BoxSize']
        particles = f['Snapshots/%d/PartType%d'%(cfg.SNAP_IDX, cfg.PART_TYPES['TNG'])]
        coords = particles['Coordinates'][s:s+l].astype(np.float32)
        e = particles['InternalEnergy'][s:s+l]
        x = particles['ElectronAbundance'][s:s+l]
        d = particles['Density'][s:s+l]
        SFR = particles['StarFormationRate'][s:s+l]

    XH = 0.76
    gamma = 5.0/3.0
    Pth = 2.0 * (1+XH) / (1 + 3*XH + 4*XH*x) * (gamma - 1) * d * e

    dcoords = coords - h.pos_TNG
    dcoords[dcoords > +0.5*BoxSize] -= BoxSize
    dcoords[dcoords < -0.5*BoxSize] += BoxSize
    r = np.linalg.norm(dcoords, axis=-1)
    del dcoords

    # compute the local standard deviations and means
    Nrbins = 100
    sorter = np.argsort(r)
    rsorted = r[sorter]
    redges = np.zeros(Nrbins+1)
    r_per_bin = len(r) // Nrbins
    for rr in range(Nrbins) :
        redges[rr+1] = rsorted[r_per_bin * rr if rr < Nrbins-1 else -1]

    indices = np.digitize(r, redges, right=True) - 1
    assert np.all(indices >= 0)
    assert np.all(indices < Nrbins)

    _, unique_counts = np.unique(indices, return_counts=True)
    print(unique_counts)

    std = np.empty(len(r))
    avg = np.empty(len(r))

    for rr in range(Nrbins) :
        # use MAD here to avoid outlier effects
        std[indices==rr] = median_abs_deviation(Pth[indices==rr])
        # use median here to avoid outlier effects
        avg[indices==rr] = np.median(Pth[indices==rr])

    del indices

    mask = (SFR == 0) & ( np.fabs(Pth - avg) < 10*std )
    print('removed ', 100 * (len(mask) - np.count_nonzero(mask)) / len(mask), ' percent')
    coords = coords[mask]
    Pth = Pth[mask]

    np.savez(h.storage_TNG, coords=coords, Pth=Pth.astype(np.float32))
