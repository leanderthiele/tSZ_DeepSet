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

fig, ax = plt.subplots(ncols=3, nrows=2)
ax_target = ax[0]
ax_prediction = ax[1]

for ii in range(3) :
    ax_target[ii].matshow(np.sum(box_target, axis=ii))    
    ax_prediction[ii].matshow(np.sum(box_prediction, axis=ii))

plt.show()
