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

ID = argv[1].replace('testing_', '').replace('.sbatch', '')
halo_idx = int(argv[2])

path_target = cfg._STORAGE_FILES['TNG']%(halo_idx, 'box_%d_cube_Pth'%cfg.TNG_RESOLUTION)
path_prediction = os.path.join(cfg.RESULTS_PATH,
                               'predictions_testing_%s_idx%d.bin'%(ID, halo_idx))
path_gaussian_prediction = os.path.join(cfg.RESULTS_PATH,
                                        'predictions_testing_%s_idx%d_seed%%d.bin'%(ID, halo_idx))

box_target = np.fromfile(path_target, dtype=np.float32)
N = cfg.TNG_RESOLUTION
x = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2].transpose(1,2,3,0).reshape((N*N*N, 3)).astype(np.float32)
x /= N # now in the range [-0.5, 0.5]
r = np.linalg.norm(x, axis=-1).flatten()
mask = r > cfg.RMAX / 5.0
box_target[mask] = 0
box_target = box_target.reshape((N, N, N))

box_prediction = CubifyPrediction(path_prediction)

seed = 0
box_gaussian_prediction = []
while True :
    if not os.path.isfile(path_gaussian_prediction%seed) :
        break
    box_gaussian_prediction.append(CubifyPrediction(path_gaussian_prediction%seed))
    seed += 1

if cfg.SCALE_PTH :
    halo_globals = eval(open(cfg._STORAGE_FILES['TNG']%(halo_idx, 'globals'), 'r').read())
    P200c = 100 * cfg._G_NEWTON * cfg._RHO_CRIT * cfg._OMEGA_B * halo_globals['M200c'] \
            / cfg._OMEGA_M / halo_globals['R200c']
    box_target /= P200c

fig, ax = plt.subplots(ncols=3, nrows=2+len(box_gaussian_prediction))
ax_target = ax[0]
ax_prediction = ax[1]

for ii in range(3) :
    vmin = 0
    vmax = np.max(np.log(np.sum(box_target, axis=ii)))
    ax_target[ii].matshow(np.log(np.sum(box_target, axis=ii)), vmin=vmin, vmax=vmax)    
    ax_prediction[ii].matshow(np.log(np.sum(box_prediction, axis=ii)), vmin=vmin, vmax=vmax)
    for jj, bgp in enumerate(box_gaussian_prediction) :
        ax[2+jj][ii].matshow(np.log(np.sum(bgp, axis=ii)), vmin=vmin, vmax=vmax)

plt.show()
