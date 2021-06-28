import numpy as np
from scipy.stats import median_abs_deviation

PLOT = False

if PLOT :
    from matplotlib import pyplot as plt
    from sys import argv
    start_idx = int(argv[1])
else :
    start_idx = 0

import h5py

from halo import Halo
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

for ii in range(start_idx, len(halo_catalog['idx_DM'])) :
    
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

    # compute thermal pressure
    XH = 0.76
    gamma = 5.0/3.0
    Pth = 2.0 * (1+XH) / (1 + 3*XH + 4*XH*x) * (gamma - 1) * d * e

    del e
    del x
    del d

    # compute radial coordinates
    dcoords = coords - h.pos_TNG
    dcoords[dcoords > +0.5*BoxSize] -= BoxSize
    dcoords[dcoords < -0.5*BoxSize] += BoxSize
    r = np.linalg.norm(dcoords, axis=-1)
    del dcoords

    # remove star forming particles
    mask = SFR==0

    if PLOT :
        r_SFR = r[~mask]
        Pth_SFR = Pth[~mask]

    r = r[mask]
    coords = coords[mask]
    Pth = Pth[mask]

    # number of particles removed due to star formation
    N_SFR = len(mask) - np.count_nonzero(mask)

    del SFR
    del mask

    # compute the local standard deviations and means
    Nrbins = 100
    rsorted = np.sort(r)
    redges = np.zeros(Nrbins+1)
    r_per_bin = len(r) // Nrbins
    for rr in range(Nrbins) :
        redges[rr+1] = rsorted[r_per_bin * (rr+1) if rr < Nrbins-1 else -1]

    del rsorted

    indices = np.digitize(r, redges, right=True) - 1
    assert np.all(indices >= 0)
    assert np.all(indices < Nrbins)
    
    ul = np.empty(len(r))

    for rr in range(Nrbins) :
        ul[indices==rr] = np.percentile(Pth[indices==rr], 98)

    del indices

    mask = (Pth<2*ul) | (r<0.01*h.R200c_DM)
    print('removed ', 100 * (len(mask) - np.count_nonzero(mask) + N_SFR) / l, ' percent')

    if PLOT :
        plt.scatter(r[mask], Pth[mask], c='black', s=0.1, label='kept')
        plt.scatter(r[~mask], Pth[~mask], c='cyan', s=0.1, label='discarded as outliers')
        plt.scatter(r_SFR, Pth_SFR, c='magenta', s=0.1, label='discarded due to SFR')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='lower left')
        plt.show()

    coords = coords[mask]
    Pth = Pth[mask]

    np.savez(h.storage_TNG, coords=coords, Pth=Pth.astype(np.float32))
