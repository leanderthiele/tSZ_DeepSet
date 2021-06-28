"""
This script produces the file halo_catalog.npz which is a useful
compression of the data we need for training.

Since we are now using the Rockstar halos, the pre-filtering of halos
has already been done in rockstar_halos/collect_particles.cpp,
so the previous filtering function of this script no longer applies.

However, it is still useful to collect some global properties of our halos.

The output halo_catalog.npz file has the fields
[direct rockstar output]
    M200c
    R200c
    pos
    ang_momentum [3]

[from collect_particles.cpp]
    min_pot_pos_DM
    min_pot_pos_TNG

[calculated here]
    inertia [3,3]
    ang_momentum2 [3] -- calculated here, could be interesting to compare to rockstar output
    vel_dispersion [3,3]

[added later by create_Nprt.py]
    Nprt_DM
    Nprt_TNG
"""

import numpy as np

import cfg

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


out = dict(M200c=[], R200c=[], Xoff=[], Voff=[],
           pos=[], ang_momentum=[],
           min_pot_pos_DM=[], min_pot_pos_TNG=[],
           inertia=[], ang_momentum2=[], vel_dispersion=[])

idx = 0

while True :
    
    print(idx)

    try :
        globals_DM = np.fromfile(cfg.STORAGE_FILES['DM']%(idx, 'globals'), dtype=np.float32)
    except FileNotFoundError :
        print('Found %d halos.'%idx)
        break

    out['M200c'].append(globals_DM[0])
    out['R200c'].append(globals_DM[1])
    out['Xoff'].append(globals_DM[2])
    out['Voff'].append(globals_DM[3])
    out['pos'].append(globals_DM[4:7])
    out['min_pot_pos_DM'].append(globals_DM[7:10])
    out['ang_momentum'].append(globals_DM[10:13])

    globals_TNG = np.fromfile(cfg.STORAGE_FILES['TNG']%(idx, 'globals'), dtype=np.float32)

    assert out['M200c'][-1] == globals_TNG[0]
    assert out['R200c'][-1] == globals_TNG[1]
    assert out['Xoff'][-1] == globals_TNG[2]
    assert out['Voff'][-1] == globals_TNG[3]
    assert np.allclose(out['pos'][-1], globals_TNG[4:7])
    out['min_pot_pos_TNG'].append(globals_TNG[7:10])
    assert np.allclose(out['ang_momentum'][-1], globals_TNG[10:13])

    # only work with DM now, as these are fields we want to use as input

    # load coordinates from file
    x = np.fromfile(cfg.STORAGE_FILES['DM']%(idx, 'coords'), dtype=np.float32)
    x = x.reshape((len(x)//3, 3))

    # load velocities from file
    v = np.fromfile(cfg.STORAGE_FILES['DM']%(idx, 'velocities'), dtype=np.float32)
    v = v.reshape((len(v)//3, 3))

    # center the coordinates
    x -= out['pos'][-1]
    x[x > +0.5*cfg.BOX_SIZE] -= cfg.BOX_SIZE
    x[x < -0.5*cfg.BOX_SIZE] += cfg.BOX_SIZE

    # remove monopole from velocities
    v -= np.mean(v, axis=0)

    out['inertia'].append(inertia(x))
    out['ang_momentum2'].append(ang_momentum(x, v))
    out['vel_dispersion'].append(inertia(v))

    idx += 1

# group as numpy arrays
for k, v in out.items() :
    out[k] = np.array(v)

# add number of halos to file
out['Nobjects'] = idx

# save to file
np.savez(cfg.HALO_CATALOG, **out)
