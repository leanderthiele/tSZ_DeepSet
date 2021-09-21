import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from halo import Halo
import paper_plot_cartoon_cfg as ctn
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

halo = Halo(halo_catalog, ctn.HALO_IDX)

N = 200 # number of pixels in image

x = np.mgrid[-N//2:N//2, -N//2:N//2].transpose(1,2,0).reshape((N, N, 2)).astype(np.float32)
x *= 2 * cfg.RMAX / N # now in the range [-2.0, 2.0]
r = np.linalg.norm(x, axis=-1)

# GNFW model
A_P0 = 3.9183
am_P0 = 0.5705
A_xc = 2.8859
am_xc = -0.8130
A_beta = 13.8758
am_beta = -0.6282

P0 = A_P0 * (0.7 * halo.M200c / 1e4)**am_P0
xc = A_xc * (0.7 * halo.M200c / 1e4)**am_xc
beta = A_beta * (0.7 * halo.M200c / 1e4)**am_beta

# be careful to do the binning identically
img = P0 * ((r+1e-3)/xc)**(-0.3) * (1 + (r+1e-3)/xc)**(-beta)

img[r > cfg.RMAX] = np.NaN


fig_b12, ax_b12 = plt.subplots(figsize=ctn.FIGSIZE)

ax_b12.matshow(np.log(img[1:, 1:]), vmin=-8.2964525, vmax=1.32657490,
               extent=(-cfg.RMAX, cfg.RMAX, -cfg.RMAX, cfg.RMAX))
circle_center = (-0.3, 0.3)
circle_kwargs = dict(facecolor='none', edgecolor='black', linestyle='dashed', linewidth=0.3)
ax_b12.add_artist(Circle(circle_center, radius=1.65, **circle_kwargs))
ax_b12.add_artist(Circle(circle_center, radius=0.9, **circle_kwargs))
ax_b12.add_artist(Circle(circle_center, radius=0.2, **circle_kwargs))
ax_b12.arrow(*circle_center, *(-x-np.sign(x)*cfg.RMAX/N for x in circle_center), width=5e-2, head_width=15e-2,
             length_includes_head=True, facecolor='black', edgecolor='none', fill=True)

ax_b12.set_frame_on(False)
ax_b12.set_xticks([])
ax_b12.set_yticks([])

fig_b12.savefig('cartoon_b12.png', bbox_inches='tight', transparent=True, pad_inches=0, dpi=1000)
