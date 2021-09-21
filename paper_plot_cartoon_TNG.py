import numpy as np
from matplotlib import pyplot as plt

from halo import Halo
import paper_plot_cartoon_cfg as ctn
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

halo = Halo(halo_catalog, ctn.HALO_IDX)

# load the target field
N = cfg.TNG_RESOLUTION
p = np.fromfile(cfg._STORAGE_FILES['TNG']%(ctn.HALO_IDX, 'box_%d_cube_Pth'%N), dtype=np.float32)
p = p.reshape((N, N, N))

x = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2].transpose(1,2,3,0).reshape((N, N, N, 3)).astype(np.float32)
x *= 5.0 / N # now in the range [-2.5, 2.5]
r = np.linalg.norm(x, axis=-1)
p[r > cfg.RMAX] = np.NaN

# restrict to RMAX
idx_max = int(N/2 * cfg.RMAX/2.5)

p = p[N//2-idx_max+1:N//2+idx_max, N//2-idx_max+1:N//2+idx_max, N//2-idx_max+1:N//2+idx_max]

fig_target, ax_target = plt.subplots(figsize=ctn.FIGSIZE)

print(np.nanmin(np.log(p[:, :, N//2].T)))
print(np.nanmax(np.log(p[:, :, N//2].T)))

ax_target.matshow(np.log(p[:, :, p.shape[2]//2].T), origin='lower', extent=(-cfg.RMAX, cfg.RMAX, -cfg.RMAX, cfg.RMAX))
ax_target.scatter([ctn.LOCAL_POS[0],], [ctn.LOCAL_POS[1],], s=10, marker='x', c='red', linewidth=ctn.MARKER_WIDTH)

ax_target.set_frame_on(False)
ax_target.set_xticks([])
ax_target.set_yticks([])

fig_target.savefig('cartoon_target.png', bbox_inches='tight', transparent=True, pad_inches=0, dpi=1000)
