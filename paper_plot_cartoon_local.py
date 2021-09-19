import numpy as np
from matplotlib import pyplot as plt

from halo import Halo
import cfg

# the halo we use as an example
HALO_IDX = 137

# where we have our example local evaluation
# (set z=0 always)
LOCAL_POS = np.array([0.5, 0.5])

# thickness in z-direction
SLAB_DEPTH = 0.05


halo_catalog = dict(np.load(cfg.HALO_CATALOG))

halo = Halo(halo_catalog, HALO_IDX)

# some useful output
Nprt = getattr(halo, 'Nprt_%d_DM'%cfg.TNG_RESOLUTION)
print('Nprt = %d'%Nprt)

# load the coordinates
x = np.fromfile(halo.storage_DM['coords'], dtype=np.float32)
x = x.reshape((len(x) // 3, 3))

assert len(x) == Nprt

# normalize the coordinates
x -= halo.pos

# periodic boundary conditions
x[x > +0.5*cfg._BOX_SIZE] -= cfg._BOX_SIZE
x[x < -0.5*cfg._BOX_SIZE] += cfg._BOX_SIZE

# normalize by R200
x /= halo.R200c

# choose particles and remove z-direction
x = x[np.fabs(x[:, 2]) < SLAB_DEPTH][:, :2]

# choose particles inside and outside of local area
mask = np.linalg.norm(x-LOCAL_POS, axis=-1) < (310.0 / halo.R200c)
x_in = x[mask]
x_out = x[~mask]

# now do the plotting
fig, ax = plt.subplots()

ax.scatter(x_in[:, 0], x_in[:, 1], c='red')
ax.scatter(x_out[:, 0], x_out[:, 1], c='black')

plt.show()
