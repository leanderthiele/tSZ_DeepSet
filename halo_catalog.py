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

[to potentially exclude halos]
    dist_ratio

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

sim_files = dict(DM = '/tigress/lthiele/Illustris_300-1_Dark/simulation.hdf5',
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
        # NOTE for some reason the multi-dimensional arrays get a spurious 0th axis
        #      of unit length which we need to remove
        out[key('pos')] = np.squeeze(grp_cat['GroupPos'][...][idx,:])
        out[key('CM')] = np.squeeze(grp_cat['GroupCM'][...][idx,:])
        out[key('prt_len')] = grp_cat['GroupLenType'][...][idx, PartType[sim_type]]
        out[key('prt_start')] = f['Offsets/%d/Group/SnapByType'%snap_idx][...][idx, PartType[sim_type]]

    return out


def match_halos(pos_DM, M200c_DM, R200c_DM, pos_type='CM', plot_dist_hist=False) :
    """
    returns the indices of the TNG halos that match the DM halos
    and the fractional distances between the DM and TNG halos in units of R200c_DM
    (can be used to exclude certain halos where the displacement is large)

    Pass R200c_DM to see a histogram of the fractional displacement
    between the DM and TNG halos in units of R200c
    (this is a useful cross-check to see if all halos have been matched)
    """
    assert pos_type in ['CM', 'Pos']
    with h5py.File(sim_files['TNG'], 'r') as f :
        boxsize = f['Parameters'].attrs['BoxSize']
        catalog = f['Groups/%d/Group'%snap_idx]
        pos_TNG = catalog['Group%s'%pos_type][...]
        M200c_TNG = catalog['Group_M_Crit200'][...]

    # first we filter out all the low-mass junk
    idx_high_mass = (M200c_TNG > 0.5 * M200c_min).nonzero()[0]
    pos_TNG = np.squeeze(pos_TNG[idx_high_mass, :])
    M200c_TNG = M200c_TNG[idx_high_mass]

    # compute all mutual distances, shape ~ [DM, TNG], taking periodic boundary conds into account
    delta_x = pos_DM[:, None,:] - pos_TNG[None, :,:]
    delta_x[delta_x > +0.5*boxsize] -= boxsize
    delta_x[delta_x < -0.5*boxsize] += boxsize
    dist = np.sqrt(np.sum(np.square(delta_x), axis=-1))

    idx_dist_min = np.argmin(dist, axis=1)
    
    dist_ratio = dist[np.arange(pos_DM.shape[0]), idx_dist_min]/R200c_DM
    # FIXME for debugging purposes
    if plot_dist_hist :
        plt.hist(dist_ratio)
        plt.show()

    return idx_high_mass[idx_dist_min], dist_ratio
    


with h5py.File(sim_files['DM'], 'r') as f :
    catalog = f['Groups/%d/Group'%snap_idx]
    M200c_DM = catalog['Group_M_Crit200'][...]
    idx_DM = (M200c_DM > M200c_min).nonzero()[0]

DM_data = get_properties(idx_DM, 'DM')

# NOTE I checked that we get identical halo matches whether we use CM or Pos here,
#      at least for the minimum mass 1e4
idx_TNG, dist_ratio = match_halos(DM_data['CM_DM'], DM_data['M200c_DM'],
                                  DM_data['R200c_DM'], plot_dist_hist=False, pos_type='CM')

TNG_data = get_properties(idx_TNG, 'TNG')

np.savez('halo_catalog.npz', M200c_min=M200c_min, snap_idx=snap_idx, dist_ratio=dist_ratio,
         **DM_data, **TNG_data)
