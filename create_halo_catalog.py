"""
This script produces the file halo_catalog.npz which is a useful
compression of the data we need for training.

Since we are now using the Rockstar halos, the pre-filtering of halos
has already been done in rockstar_halos/collect_particles.cpp,
so the previous filtering function of this script no longer applies.

However, it is still useful to collect some global properties of our halos.
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


# this is a bit repetitive as we could infer these from other places,
# but it is useful to have them centrally available in this script
out = dict(M200c=[], R200c=[], Xoff=[], Voff=[],
           Vmax=[], Vrms=[], Rs=[], rs_klypin=[],
           M200c_all=[], Mvir=[], M200b=[], M500c=[],
           M2500c=[], Spin=[], spin_bullock=[],
           b_to_a=[], c_to_a=[],
           pos=[], vel=[], ang_mom=[],
           min_pot_pos_DM=[], min_pot_pos_TNG=[],
           inertia=[], CM=[], ang_mom2=[], vel_dispersion=[])

idx = 0

while True :
    
    print(idx)

    try :
        globals_DM = eval(open(cfg.STORAGE_FILES['DM']%(idx, 'globals'), 'r').readline())
    except FileNotFoundError :
        print('Found %d halos.'%idx)
        break

    assert isinstance(globals_DM, dict)

    for k, v in globals_DM.items() :
        out[k].append(v)

    globals_TNG = eval(open(cfg.STORAGE_FILES['TNG']%(idx, 'globals'), 'r').readline())
    assert isinstance(globals_TNG, dict)

    out['min_pot_pos_TNG'].append(globals_TNG.pop('min_pot_pos_TNG'))

    for k, v in globals_TNG.items() :
        if isinstance(v, float) :
            assert out[k][-1] == v
        elif isinstance(v, np.ndarray) :
            assert np.allclose(out[k][-1], v)
        else :
            raise RuntimeError('%s has unexpected type'%k)

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
    out['CM'].append(np.mean(x, axis=0))
    out['ang_mom2'].append(ang_momentum(x, v))
    out['vel_dispersion'].append(inertia(v))

    idx += 1

# group as numpy arrays
for k, v in out.items() :
    out[k] = np.array(v, dtype=np.float32)

# add number of halos to file
out['Nobjects'] = idx

# save to file
np.savez(cfg.HALO_CATALOG, **out)
