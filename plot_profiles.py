"""
Command line arguments:
    [1] ID
    [2] halo idx
"""

from sys import argv
import os.path

import numpy as np
from matplotlib import pyplot as plt

from cubify_prediction import CubifyPrediction
import cfg

RMIN = 0.0
RMAX = 2.0
NR = 20
RBINS = np.linspace(RMIN, RMAX, num=NR+1)
RCENTERS = 0.5 * (RBINS[1:] + RBINS[:-1])

ID = argv[1].replace('testing_', '').replace('.sbatch', '')
halo_idx = int(argv[2])

N = cfg.TNG_RESOLUTION
x = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2].transpose(1,2,3,0).reshape((N*N*N, 3)).astype(np.float32)
x *= 5.0 / N # now in the range [-2.5, 2.5]
r = np.linalg.norm(x, axis=-1).flatten()

path_target = cfg._STORAGE_FILES['TNG']%(halo_idx, 'box_%d_cube_Pth'%cfg.TNG_RESOLUTION)
path_prediction = os.path.join(cfg.RESULTS_PATH,
                               'predictions_testing_%s_idx%d.bin'%(ID, halo_idx))
path_gaussian_prediction = os.path.join(cfg.RESULTS_PATH,
                                        'predictions_testing_%s_idx%d_seed%%d.bin'%(ID, halo_idx))

box_target = np.fromfile(path_target, dtype=np.float32)

box_prediction = CubifyPrediction(path_prediction).flatten()

seed = 0
box_gaussian_prediction = []
while True :
    if not os.path.isfile(path_gaussian_prediction%seed) :
        break
    box_gaussian_prediction.append(CubifyPrediction(path_gaussian_prediction%seed).flatten())
    seed += 1

halo_globals = eval(open(cfg._STORAGE_FILES['TNG']%(halo_idx, 'globals'), 'r').read())
P200c = 100 * cfg._G_NEWTON * cfg._RHO_CRIT * cfg._OMEGA_B * halo_globals['M200c'] \
        / cfg._OMEGA_M / halo_globals['R200c']
box_target /= P200c

if not cfg.SCALE_PTH :
    box_prediction /= P200c
    for bgp in box_gaussian_prediction :
        bgp /= P200c

# GNFW model
A_P0 = 3.9183
am_P0 = 0.5705
A_xc = 2.8859
am_xc = -0.8130
A_beta = 13.8758
am_beta = -0.6282

P0 = A_P0 * (0.7 * halo_globals['M200c'] / 1e4)**am_P0
xc = A_xc * (0.7 * halo_globals['M200c'] / 1e4)**am_xc
beta = A_beta * (0.7 * halo_globals['M200c'] / 1e4)**am_beta

# be careful to do the binning identically
box_b12 = P0 * ((r+1e-3)/xc)**(-0.3) * (1 + (r+1e-3)/xc)**(-beta)

def get_binned(radii, values) :
    assert len(radii) == len(values)
    indices = np.digitize(radii, RBINS) - 1
    out = np.zeros(NR)
    for ii in range(NR) :
        out[ii] = np.mean(values[indices==ii])
    return out

target_binned = get_binned(r, box_target)
prediction_binned = get_binned(r, box_prediction)
gaussian_prediction_binned = [get_binned(r, bgp) for bgp in box_gaussian_prediction]
b12_binned = get_binned(r, box_b12)


fig, ax = plt.subplots()

ax.plot(RCENTERS, target_binned, label='target')
ax.plot(RCENTERS, prediction_binned, label='reconstruction')
ax.plot(RCENTERS, b12_binned, label='GNFW')
for ii, gp in enumerate(gaussian_prediction_binned) :
    ax.plot(RCENTERS, gp, label='gauss #%d'%ii, linestyle='dashed')

ax.set_yscale('log')
ax.set_xlim(RMIN, RMAX)
ax.set_xlabel('R/R200')
ax.set_ylabel('P/P200')
ax.legend(loc='upper right')

plt.show()
