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
fig_local, ax_local = plt.subplots(figsize=ctn.FIGSIZE)
fig_origin, ax_origin = plt.subplots(figsize=ctn.FIGSIZE)

ax_local.scatter(x_out[:, 0], x_out[:, 1], c='grey', s=ctn.MARKER_SIZE)
ax_local.scatter(x_in[:, 0], x_in[:, 1], c='green', s=ctn.MARKER_SIZE)
ax_local.scatter([ctn.LOCAL_POS[0],], [ctn.LOCAL_POS[1],], s=10, marker='x', c='red', linewidth=ctn.MARKER_WIDTH)

ax_origin.scatter(x[:, 0], x[:, 1], c='green', s=ctn.MARKER_SIZE)

ax_local.set_frame_on(False)
ax_local.set_xticks([])
ax_local.set_yticks([])

ax_origin.set_frame_on(False)
ax_origin.set_xticks([])
ax_origin.set_yticks([])

fig_local.savefig('cartoon_local.png', bbox_inches='tight', transparent=True, pad_inches=0, dpi=1000)
fig_origin.savefig('cartoon_origin.png', bbox_inches='tight', transparent=True, pad_inches=0, dpi=1000)
