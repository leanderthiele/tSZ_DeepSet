PLOT = False
RECOMPUTE = False

import os.path
from glob import glob

import numpy as np

from matplotlib import pyplot as plt

NBINS = 20

ROOT = '/scratch/gpfs/lthiele/tSZ_DeepSet_pca'

FNAMES = glob(os.path.join(ROOT, '*.npz'))

print(len(FNAMES))

RBINS = np.linspace(0, 1.5, num=NBINS+1)
RCENTERS = 0.5*(RBINS[:-1] + RBINS[1:])

def get_binned(x, indices) :
    out = np.empty(NBINS)
    for ii in range(NBINS) :
        out[ii] = np.mean(x[indices==ii])
    return out

if RECOMPUTE or not os.path.isfile(os.path.join(ROOT, 'data.npy')) :
    data = np.empty((len(FNAMES), NBINS))

    for ff, fname in enumerate(FNAMES) :

        print(fname)

        with np.load(fname) as f :
            r = f['r']
            p = f['prediction']
            t = f['target']

        sorter = np.argsort(r)
        r = r[sorter]
        p = p[sorter]
        t = t[sorter]

        if PLOT :
            print(r.shape)
            print(p.shape)
            print(t.shape)
            plt.semilogy(r, t, label='target')
            plt.semilogy(r, p, label='prediction')
            plt.legend()
            plt.show()

        indices = np.digitize(r, RBINS) - 1
        assert np.min(indices) == 0

        p_binned = get_binned(p, indices)
        t_binned = get_binned(t, indices)

        if PLOT :
            plt.semilogy(p_binned, label='prediction')
            plt.semilogy(t_binned, label='target')
            plt.legend()
            plt.show()

        data[ff, :] = (t_binned - p_binned) / p_binned

    np.save(os.path.join(ROOT, 'data.npy'), data)

else : # data file exists and RECOMPUTE is false
    data = np.load(os.path.join(ROOT, 'data.npy'))

# center
data -= np.mean(data, axis=0, keepdims=True)

# normalize
data /= np.std(data, axis=0, keepdims=True)

# compute covariance matrix
C = data.T @ data / (len(FNAMES)-1)
assert np.allclose(C, C.T)

# diagonalize
w, v = np.linalg.eigh(C)

# numpy returns eigenvalues in ascending order, descending is more convenient
w = w[::-1]
v = v.T[::-1, :]

plt.semilogy(w, marker='o')
plt.xlabel('index')
plt.ylabel('principal component score')
plt.show()

for ii in range(4) :
    plt.plot(RCENTERS, v[ii] / np.sign(v[ii,0]), label='%d (score=%.1f)'%(ii,w[ii]))
plt.xlabel('$R/R_{\sf 200c}$')
plt.ylabel('principal component')
plt.legend()
plt.show()

plt.hist(data @ v[0,:], bins=40)
plt.show()
