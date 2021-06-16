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

    inertia_DM [3,3]
    ang_momentum_DM [3]
    central_CM_DM [N, 3] -- vectors w.r.t. cfg.ORIGIN
    vel_dispersion_DM [3, 3]

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

# whether to create the compute-intensive inertia and angular momentum fields
# (set to false for some quick output that cannot be used for training)
DO_LONG = True

import numpy as np
import h5py

import cfg

# lower mass cutoff -- in code units, i.e. 1e10 Msun/h
# this refers to the DM-only mass
M200c_min = 5e3

# for which radii around pos, in units of R200c, we want to compute the CMs
central_CM_radii = [0.1, 0.3, 1.0]

out_file = cfg.HALO_CATALOG if DO_LONG else 'short_'+cfg.HALO_CATALOG

try :
    with np.load(out_file) as f :
        if abs(f['M200c_min']/M200c_min - 1) < 1e-5 \
           and f['snap_idx'] == cfg.SNAP_IDX :
            print('Consistent file %s already exists. Aborting.'%out_file)
            exit()
except FileNotFoundError :
    print('Did not find existing %s, will compute.'%out_file)


def inertia(x) :
    rsq_sum = np.einsum('ni,ni->', x, x)
    xij_sum = np.einsum('ni,nj->ij', x, x)
    T = rsq_sum * np.identity(3) - xij_sum
    return T


def ang_momentum(x, v) :
    M = np.sum(np.cross(x, v), axis=0)
    return M


def central_CM(r, x, rmax) :
    xc = x[r < rmax]
    return np.sum(xc, axis=0) / xc.shape[0]


def get_properties(idx, sim_type) :
    """ sim_type is either DM or TNG """
    out = dict()

    key = lambda s : '%s_%s'%(s, sim_type)

    with h5py.File(cfg.SIM_FILES[sim_type], 'r') as f :
        grp_cat = f['Groups/%d/Group'%cfg.SNAP_IDX]
        out[key('idx')] = idx
        out[key('M200c')] = grp_cat['Group_M_Crit200'][...][idx]
        out[key('R200c')] = grp_cat['Group_R_Crit200'][...][idx]
        # NOTE for some reason the multi-dimensional arrays get a spurious 0th axis
        #      of unit length which we need to remove
        out[key('pos')] = np.squeeze(grp_cat['GroupPos'][...][idx,:])
        out[key('CM')] = np.squeeze(grp_cat['GroupCM'][...][idx,:])
        out[key('prt_len')] = grp_cat['GroupLenType'][...][idx, cfg.PART_TYPES[sim_type]]
        out[key('prt_start')] = f['Offsets/%d/Group/SnapByType'%cfg.SNAP_IDX][...][idx, cfg.PART_TYPES[sim_type]]

    if sim_type == 'DM' and DO_LONG :
        out[key('inertia')] = np.empty((len(out[key('idx')]), 3, 3))
        out[key('ang_momentum')] = np.empty((len(out[key('idx')]), 3))
        out[key('central_CM')] = np.empty((len(out[key('idx')]), len(central_CM_radii), 3))
        
        with h5py.File(cfg.SIM_FILES[sim_type], 'r') as f :
            for ii in range(len(out[key('idx')])) :
                print('Computing inertia, ang_momentum, central_CM for %d'%ii)
                # compute inertia tensor and angular momentum
                particles = f['Snapshots/%d/PartType%d'%(cfg.SNAP_IDX, cfg.PART_TYPES[sim_type])]
                _s = out[key('prt_start')][ii]
                _l = out[key('prt_len')][ii]

                # do the intermediate summations in double precision to minimize roundoff error
                # we also save the arrays in double precision
                coords = particles['Coordinates'][_s : _s+_l].astype(np.float64)
                velocities = particles['Velocities'][_s : _s+_l].astype(np.float64)

                coords_pos = coords - out[key('pos')][ii,:].astype(np.float64)
                coords_pos[coords_pos > +0.5*cfg.BOX_SIZE] -= cfg.BOX_SIZE
                coords_pos[coords_pos < -0.5*cfg.BOX_SIZE] += cfg.BOX_SIZE
                r_pos = np.linalg.norm(coords_pos, axis=-1)
                del coords_pos

                coords_CM = coords - out[key('CM')][ii,:].astype(np.float64)
                coords_CM[coords_CM > +0.5*cfg.BOX_SIZE] -= cfg.BOX_SIZE
                coords_CM[coords_CM < -0.5*cfg.BOX_SIZE] += cfg.BOX_SIZE

                for jj in range(len(central_CM_radii)) :
                    out[key('central_CM')][ii,jj,:] = central_CM(r_pos, coords_CM,
                                                                 central_CM_radii[jj] * out[key('R200c')][ii])
                del coords_CM
                del r_pos

                coords_centered = coords - out[key(cfg.ORIGIN)][ii,:].astype(np.float64)
                coords_centered[coords_centered > +0.5*cfg.BOX_SIZE] -= cfg.BOX_SIZE
                coords_centered[coords_centered < -0.5*cfg.BOX_SIZE] += cfg.BOX_SIZE

                out[key('inertia')][ii, ...] = inertia(coords_centered)
                out[key('ang_momentum')][ii, ...] = ang_momentum(coords_centered, velocities)
                del coords_centered

                velocities -= np.mean(velocities, axis=0)
                out[key('vel_dispersion')] = inertia(velocities)

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
    with h5py.File(cfg.SIM_FILES['TNG'], 'r') as f :
        boxsize = f['Parameters'].attrs['BoxSize']
        catalog = f['Groups/%d/Group'%cfg.SNAP_IDX]
        pos_TNG = catalog['Group%s'%pos_type][...]
        M200c_TNG = catalog['Group_M_Crit200'][...]

    # first we filter out all the low-mass junk
    idx_high_mass = (M200c_TNG > 0.2 * M200c_min).nonzero()[0]
    pos_TNG = np.squeeze(pos_TNG[idx_high_mass, :])
    M200c_TNG = M200c_TNG[idx_high_mass]

    # compute all mutual distances, shape ~ [DM, TNG], taking periodic boundary conds into account
    delta_x = pos_DM[:, None,:] - pos_TNG[None, :,:]
    delta_x[delta_x > +0.5*boxsize] -= boxsize
    delta_x[delta_x < -0.5*boxsize] += boxsize
    dist = np.sqrt(np.sum(np.square(delta_x), axis=-1))

    idx_dist_min = np.argmin(dist, axis=1)
    
    dist_ratio = dist[np.arange(pos_DM.shape[0]), idx_dist_min]/R200c_DM

    return idx_high_mass[idx_dist_min], dist_ratio
    


with h5py.File(cfg.SIM_FILES['DM'], 'r') as f :
    catalog = f['Groups/%d/Group'%cfg.SNAP_IDX]
    M200c_DM = catalog['Group_M_Crit200'][...]
    idx_DM = (M200c_DM > M200c_min).nonzero()[0]

DM_data = get_properties(idx_DM, 'DM')

# NOTE I checked that we get identical halo matches whether we use CM or Pos here,
#      at least for the minimum mass 1e4
idx_TNG, dist_ratio = match_halos(DM_data['CM_DM'], DM_data['M200c_DM'],
                                  DM_data['R200c_DM'], plot_dist_hist=False, pos_type='CM')

TNG_data = get_properties(idx_TNG, 'TNG')

np.savez(out_file, M200c_min=M200c_min, snap_idx=cfg.SNAP_IDX, dist_ratio=dist_ratio,
         **DM_data, **TNG_data)
