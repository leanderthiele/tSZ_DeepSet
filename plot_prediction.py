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

path = os.path.join(cfg.RESULTS_PATH, 'prediction_%s_idx%d.bin'%(ID, halo_idx))

box = CubifyPrediction(path)

fig, ax = plt.subplots(ncols=3)

for ii in range(3) :
    ax[ii].matshow(np.sum(box, axis=ii))

plt.show()
