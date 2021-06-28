PLOT = False

import numpy as np

if PLOT :
    from matplotlib import pyplot as plt
    from sys import argv
    start_idx = int(argv[1])
else :
    start_idx = 0

from halo import Halo
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

for ii in range(start_idx, halo_catalog['Nobjects']) :

    print(ii)

    h = Halo(halo_catalog, ii)

    # load the TNG coordinates
    x = np.fromfile(h.storage_TNG['coords'], dtype=np.float32)
    x = x.reshape((len(x)//3, 3))

    # compute radial coordinates
    # We have checked that the Rockstar position generally gives better results
    # than the position of the TNG particle with minimum potential energy
    dx = x - h.pos
    dx[dx > +0.5*cfg.BOX_SIZE] -= cfg.BOX_SIZE
    dx[dx < -0.5*cfg.BOX_SIZE] += cfg.BOX_SIZE
    r = np.linalg.norm(dx, axis=-1)
    del dx

    # generate the radial bins
    Nrbins = 100
    rsorted = np.sort(r)
    redges = np.zeros(Nrbins+1)
    r_per_bin = len(r) // Nrbins
    for rr in range(Nrbins) :
        redges[rr+1] = rsorted[r_per_bin * (rr+1)] if rr < Nrbins-1 \
                       else (1+1e-8) * rsorted[-1]
    del rsorted

    # find in which radial bin each TNG particle falls
    indices = np.digitize(r, redges, right=False) - 1
    assert np.all(indices >= 0)
    assert np.all(indices < Nrbins)

    # load the pressure
    Pth = np.fromfile(h.storage_TNG['Pth'], dtype=np.float32)

    # compute upper limits for each bin (above which we remove particles as outliers)
    ul = np.empty(len(r))
    for rr in range(Nrbins) :
        ul[indices==rr] = 2 * np.percentile(Pth[indices==rr], 98)
    del indices

    # construct our mask
    mask = (Pth < ul) | (r < 0.01*h.R200c)
    print('removed ', 100*(len(mask) - np.count_nonzero(mask)) / len(mask), ' percent')

    if PLOT :
        plt.scatter(r[mask], Pth[mask], c='black', s=0.1, label='kept')
        plt.scatter(r[~mask], Pth[~mask], c='cyan', s=0.1, label='discarded')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='lower left')
        plt.show()

    # load the TNG masses
    m = np.fromfile(h.storage_TNG['masses'], dtype=np.float32)

    x = x[mask]
    Pth = Pth[mask]
    m = m[mask]

    # save back into binary files
    x.tofile(h.storage_TNG['coords_filtered'])
    Pth.tofile(h.storage_TNG['Pth_filtered'])
    m.tofile(h.storage_TNG['masses_filtered'])
