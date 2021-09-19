import numpy as np
from matplotlib import pyplot as plt

from halo import Halo
import paper_plot_cartoon_cfg as ctn
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

halo = Halo(halo_catalog, ctn.HALO_IDX)

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
x = x[np.fabs(x[:, 2]) < ctn.SLAB_DEPTH][:, :2]

# choose particles inside and outside of local area
print(halo.R200c)
mask = np.linalg.norm(x-np.array(ctn.LOCAL_POS)[None, :], axis=-1) < (310.0 / halo.R200c)
x_in = x[mask]
x_out = x[~mask]

# now do the plotting
fig, ax = plt.subplots(figsize=(1,1))

ax.scatter(x_out[:, 0], x_out[:, 1], c='black', s=ctn.MARKER_SIZE)
ax.scatter(x_in[:, 0], x_in[:, 1], c='green', s=ctn.MARKER_SIZE)

ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])

fig.savefig('cartoon_local.png', bbox_inches='tight', transparent=True, pad_inches=0, dpi=1000)
