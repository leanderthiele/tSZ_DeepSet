import numpy as np

from halo import Halo
import paper_plot_cartoon_cfg as ctn
import cfg

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

halo = Halo(halo_catalog, ctn.HALO_IDX)

res = np.fromfile(halo.storage_TNG['residuals'], dtype=np.float32)

rbins = np.linspace(cfg.RESIDUALS_RMIN, cfg.RESIDUALS_RMAX, cfg.RESIDUALS_NBINS+1)
rcenters = 0.5 * (rbins[1:] + rbins[:-1])

np.savetxt('cartoon_residuals.dat', np.stack((rcenters, res)).T,
           delimiter=',', header='r,DeltaPth', comments='')
