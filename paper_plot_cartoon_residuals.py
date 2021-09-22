import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import interp1d

from halo import Halo
import paper_plot_cartoon_cfg as ctn
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

halo = Halo(halo_catalog, ctn.HALO_IDX)

res = np.fromfile(halo.storage_TNG['residuals'], dtype=np.float32)

rbins = np.linspace(cfg.RESIDUALS_RMIN, cfg.RESIDUALS_RMAX, cfg.RESIDUALS_NBINS+1)
rcenters = 0.5 * (rbins[1:] + rbins[:-1])

N = 50 # number of pixels in image

x = np.mgrid[-N//2:N//2, -N//2:N//2].transpose(1,2,0).reshape((N, N, 2)).astype(np.float32)
x *= 2 * cfg.RMAX / N # now in the range [-2.0, 2.0]
r = np.linalg.norm(x, axis=-1)

# interpolate the residuals and populate image
res_interp = interp1d(rcenters, res, bounds_error=False, fill_value='extrapolate')

img = res_interp(r)
img[r > cfg.RMAX] = np.NaN

fig_residuals, ax_residuals = plt.subplots(figsize=ctn.FIGSIZE)

ax_residuals.matshow(img[1:, 1:])

ax_residuals.set_frame_on(False)
ax_residuals.set_xticks([])
ax_residuals.set_yticks([])

fig_residuals.savefig('cartoon_residuals.png', bbox_inches='tight', transparent=True, pad_inches=0, dpi=1000)
