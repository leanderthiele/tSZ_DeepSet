import os.path
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

import cfg

ID = 'vae64_200epochs_usekld_onelatent_nr2074_ontestingset'

all_indices = []
for idx in range(470) :
    if os.path.isfile(os.path.join(cfg.RESULTS_PATH, 'predictions_testing_%s_idx%d.bin'%(ID, idx))) :
        all_indices.append(idx)

# logarithmic edges
edges = np.linspace(-3, 2, num=50)
centers = 0.5 * (edges[1:] + edges[:-1])

h_target = np.zeros(len(centers), dtype=int)
h_prediction = np.zeros(len(centers), dtype=int)
h_b12 = np.zeros(len(centers), dtype=int)

# filter out the points we don't have predictions for
N = cfg.TNG_RESOLUTION
x = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2].transpose(1,2,3,0).reshape((N*N*N, 3)).astype(np.float32)
x *= 5.0 / N # now in the range [-2.5, 2.5]
r = np.linalg.norm(x, axis=-1).flatten()
mask = r < cfg.RMAX
r = r[mask]

# GNFW model
A_P0 = 3.9183
am_P0 = 0.5705
A_xc = 2.8859
am_xc = -0.8130
A_beta = 13.8758
am_beta = -0.6282

fig, ax = plt.subplots(figsize=(5,3))

for ii, halo_idx in enumerate(all_indices) :
    
    print('%d / %d'%(ii, len(all_indices)))

    path_target = cfg._STORAGE_FILES['TNG']%(halo_idx, 'box_%d_cube_Pth'%cfg.TNG_RESOLUTION)
    path_prediction = os.path.join(cfg.RESULTS_PATH, 'predictions_testing_%s_idx%d.bin'%(ID, halo_idx))

    halo_globals = eval(open(cfg._STORAGE_FILES['TNG']%(halo_idx, 'globals'), 'r').read())
    M200c = halo_globals['M200c']
    R200c = halo_globals['R200c']
    P200c = 100 * cfg._G_NEWTON * cfg._RHO_CRIT * cfg._OMEGA_B * M200c \
            / cfg._OMEGA_M / R200c

    box_target = np.fromfile(path_target, dtype=np.float32)
    box_target = box_target[mask]

    P0 = A_P0 * (0.7 * M200c / 1e4)**am_P0
    xc = A_xc * (0.7 * M200c / 1e4)**am_xc
    beta = A_beta * (0.7 * M200c / 1e4)**am_beta

    # be careful to do the binning identically
    box_b12 = P0 * ((r+1e-3)/xc)**(-0.3) * (1 + (r+1e-3)/xc)**(-beta)

    if cfg.SCALE_PTH :
        box_target /= P200c
    else :
        box_b12 *= P200c

    box_prediction = np.fromfile(path_prediction, dtype=np.float32)

    h, _ = np.histogram(np.log10(box_target), bins=edges)
    h_target += h

    h, _ = np.histogram(np.log10(box_prediction), bins=edges)
    h_prediction += h

    h, _ = np.histogram(np.log10(box_b12), bins=edges)
    h_b12 += h

ax.step(centers, h_prediction, label='reconstruction', where='mid')
ax.step(centers, h_b12, label='GNFW benchmark', where='mid')
ax.scatter(centers, h_target, label='target', s=4, c='green', zorder=10)

ax.legend(loc='lower center')
ax.set_xlabel('$\log_{10}(P_e/P_{200})$')
ax.set_ylabel('counts')
ax.set_yscale('log')
ax.set_xlim(np.min(edges), np.max(edges))

fig.savefig('PDF.pdf', bbox_inches='tight')
