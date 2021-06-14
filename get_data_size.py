"""
computes the disk space needed for all simulation data
"""

import numpy as np

with np.load('halo_catalog.npz') as f :
    N_DM = np.sum(f['prt_len_DM'])
    N_TNG = np.sum(f['prt_len_TNG'])


# DM : we only need coordinates
dim_DM = 3

# TNG : we need coordinates and pressure
dim_TNG = 4

# store as 32bit floats
type_size = 4


space = (N_DM * dim_DM + N_TNG * dim_TNG) * type_size / 1024**3

print(space, 'GB')
