RECOMPUTE = True

import os.path
from glob import glob

import numpy as np

from data_modes import DataModes
import cfg

ROOT = '/scratch/gpfs/lthiele/tSZ_DeepSet_pca'

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

RBINS = np.linspace(cfg.BINNING_RMIN, cfg.BINNING_RMAX, num=cfg.BINNING_NBINS+1)
RCENTERS = 0.5*(RBINS[:-1] + RBINS[1:])

def get_binned(x, indices) :
    out = np.empty(cfg.BINNING_NBINS)
    for ii in range(cfg.BINNING_NBINS) :
        out[ii] = np.mean(x[indices==ii])
    return out

if RECOMPUTE or not os.path.isfile(os.path.join(ROOT, 'data.npy')) :

    data = np.empty((halo_catalog['Nobjects'], cfg.BINNING_NBINS))

    for ff in range(halo_catalog['Nobjects']) :

        fname = os.path.join(ROOT, '%d.npz'%ff)
        print(fname)

        with np.load(fname) as f :
            r = f['r']
            p = f['prediction']
            t = f['target']

        sorter = np.argsort(r)
        r = r[sorter]
        p = p[sorter]
        t = t[sorter]

        indices = np.digitize(r, RBINS) - 1
        assert np.min(indices) == 0

        p_binned = get_binned(p, indices)
        t_binned = get_binned(t, indices)

        data[ff, :] = (t_binned - p_binned) / p_binned

    np.save(os.path.join(ROOT, 'data.npy'), data)

else : # data file exists and RECOMPUTE is false
    data = np.load(os.path.join(ROOT, 'data.npy'))

# slice the data corresponding to the training set
training_data = data[DataModes.TRAINING.sample_indices(len(data))]

# center
data -= np.mean(data[training_data, axis=0, keepdims=True)

# normalize
data /= np.std(training_data, axis=0, keepdims=True)

# now save to files
for ii in range(len(data)) :
    data[ii].tofile(cfg._STORAGE_FILES%(ii, 'residuals'))
