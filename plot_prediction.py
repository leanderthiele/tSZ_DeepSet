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
path_prediction = os.path.join(cfg.RESULTS_PATH, 'predictions_testing_%s_idx%d.bin'%(ID, halo_idx))

box_target = np.fromfile(path_target, dtype=np.float32).reshape([cfg.TNG_RESOLUTION,]*3)
box_prediction = CubifyPrediction(path_prediction)

if cfg.SCALE_PTH :
    halo_globals = eval(open(cfg._STORAGE_FILES['TNG']%(halo_idx, 'globals'), 'r').read())
    P200c = 100 * cfg._G_NEWTON * cfg._RHO_CRIT * cfg._OMEGA_B * halo_globals['M200c'] \
            / cfg._OMEGA_M / halo_globals['R200c']
    box_target /= P200c

fig, ax = plt.subplots(ncols=3, nrows=2)
ax_target = ax[0]
ax_prediction = ax[1]

for ii in range(3) :
    vmin = 0
    vmax = np.max(np.log(np.sum(box_target, axis=ii)))
    ax_target[ii].matshow(np.log(np.sum(box_target, axis=ii)), vmin=vmin, vmax=vmax)    
    ax_prediction[ii].matshow(np.log(np.sum(box_prediction, axis=ii)), vmin=vmin, vmax=vmax)

plt.show()
