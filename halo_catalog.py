"""
This script produces the file halo_catalog.npz which is a useful
compression of the data we need for training.

The primary purpose is to extract all halos from the group catalog
that we are actually interested in, save some of their useful properties
and perform the matching between DM-only and TNG catalogs.

The output file contains the fields:

[consistency]
    M200c_min
    snap_idx

[DM-only simulation]
    idx_DM
    pos_DM
    CM_DM
    M200c_DM
    R200c_DM
    prt_start_DM
    prt_len_DM

[TNG simulation]
    idx_TNG
    pos_TNG
    CM_TNG
    M200c_TNG
    R200c_TNG
    prt_start_TNG
    prt_len_TNG

The prt_* fields in the TNG case refer to the gas particles.
"""

import numpy as np
from matplotlib import pyplot as plt
import h5py

# lower mass cutoff -- in code units, i.e. 1e10 Msun/h
# this refers to the DM-only mass
M200c_min = 1e4

# specify the redshift here
snap_idx = 99

PartType = dict(DM = 1, TNG = 0)

sim_files = dict(DM = '/tigress/lthiele/Illustris_300-1_DM/simulation.hdf5',
                 TNG = '/tigress/lthiele/Illustris_300-1_TNG/simulation.hdf5')

try :
    with np.load('halo_catalog.npz') as f :
        if abs(f['M200c_min']/M200c_min - 1) < 1e-5 \
           and f['snap_idx'] == snap_idx :
            print('Consistent file halo_catalog.npz already exists. Aborting.')
            exit()
except FileNotFoundError :
    print('Did not find existing halo_catalog.npz, will compute.')


def get_properties(idx, sim_type) :
    """ sim_type is either DM or TNG """
    out = dict()

    key = lambda s : '%s_%s'%(s, sim_type)

    with h5py.File(sim_files[sim_type], 'r') as f :
        grp_cat = f['Groups/%d/Group'%snap_idx]
        out[key('idx')] = idx
        out[key('M200c')] = grp_cat['Group_M_Crit200'][...][idx]
        out[key('R200c')] = grp_cat['Group_R_Crit200'][...][idx]
        out[key('pos')] = grp_cat['GroupPos'][...][idx,:]
        out[key('CM')] = grp_cat['GroupCM'][...][idx,:]
        out[key('prt_len')] = grp_cat['GroupLenType'][...][idx, PartType[sim_type]]
        out[key('prt_start')] = f['Offsets/%d/Group/SnapByType'%snap_idx][...][idx, PartType[sim_type]]

    return out


def match_halos(pos_DM, M200c_DM, pos_type='CM') :
    """ returns the indices of the TNG halos that match the DM halos,
        there maybe be some experimentation required here
    """
    assert pos_type in ['DM', 'Pos']
    with h5py.File(sim_files['TNG'], 'r') as f :
        catalog = f['Groups/%d/Group'%snap_idx]
        pos_TNG = catalog['Group%s'%pos_type][...]
        M200c_TNG = catalog['Group_M_Crit200'][...]

    # first we filter out all the low-mass junk
    idx_high_mass = (M200c_TNG > 0.5 * M200c_min).nonzero()
    pos_TNG = pos_TNG[idx_high_mass, :]
    M200c_TNG = M200c_TNG[idx_high_mass, :]

    # compute all mutual distances, shape ~ [DM, TNG]
    dist = np.sqrt(np.sum(np.square(pos_DM[:, None] - pos_TNG[None, :])))

    idx_dist_min = np.argmin(dist, axis=1)
    
    # FIXME for debugging purposes
    plt.hist(dist[:, idx_dist_min])
    plt.show()

    return idx_high_mass[idx_dist_min]
    


with h5py.File(sim_files['DM'], 'r') as f :
    catalog = f['Groups/%d/Group'%snap_idx]
    M200c_DM = catalog['Group_M_Crit200'][...]
    idx_DM = (M200c_DM > M200c_min).nonzero()

DM_data = get_properties(idx_DM, 'DM')

idx_TNG = match_halos(DM_data['CM_DM'], DM_data['M200c_DM'])

TNG_data= get_properties(idx_TNG, 'TNG')

np.savez('halo_catalog.npz', M200c_min=M200_min, snap_idx=snap_idx,
         **DM_data, **TNG_data)
