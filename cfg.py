
# which redshift snapshot to work with
SNAP_IDX = 99

# the particle types in Illustris
PART_TYPES = dict(DM = 1, TNG = 0)

# where we have the halo catalog stored
HALO_CATALOG = 'halo_catalog.npz'

# the (virtual) hdf5 files where we find the simulations
SIM_FILES = dict(DM = '/tigress/lthiele/Illustris_300-1_Dark/simulation.hdf5',
                 TNG = '/tigress/lthiele/Illustris_300-1_TNG/simulation.hdf5')

# how many particles of each type to use per halo
# set to None if all particles are to be used
PRT_FRACTION = dict(DM = None, TNG = None)

# whether to divide the coordinates by R200c
NORMALIZE_COORDS = True
