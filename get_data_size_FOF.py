"""
computes the disk space needed for all simulation data
pass halo catalog file as command line argument
"""

from sys import argv

import numpy as np

with np.load(argv[1]) as f :
    N_halos = len(f['idx_DM'])
    min_N_DM = np.min(f['prt_len_DM'])
    min_N_TNG = np.min(f['prt_len_TNG'])
    N_DM = np.sum(f['prt_len_DM'])
    N_TNG = np.sum(f['prt_len_TNG'])


# DM : we only need coordinates
dim_DM = 3

# TNG : we need coordinates and pressure
dim_TNG = 4

# store as 32bit floats
type_size = 4


space = (N_DM * dim_DM + N_TNG * dim_TNG) * type_size / 1024**3

print(N_halos, 'halos')
print(min_N_DM, 'minimum number of DM particles')
print(min_N_TNG, 'minimum number of TNG particles')
print(space, 'GB')
