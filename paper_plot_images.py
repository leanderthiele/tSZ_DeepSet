import os.path

import numpy as np
from matplotlib import pyplot as plt

from paper_plot_get_indices import GetIndices
from cubify_prediction import CubifyPrediction
import cfg

NPANELS = 4

ID = 'vae64_200epochs_usekld_onelatent_nr2074_ontestingset'

RNG = np.random.default_rng(137)

halo_indices = GetIndices(NPANELS)

fig, ax = plt.subplots(ncols=NPANELS, nrows=3,
                       figsize=(5,4), gridspec_kw={'hspace': 0, 'wspace': 0})
ax = ax.T

for a, halo_idx in zip(ax, halo_indices) :
    path_target = cfg._STORAGE_FILES['TNG']%(halo_idx, 'box_%d_cube_Pth'%cfg.TNG_RESOLUTION)
    path_prediction = os.path.join(cfg.RESULTS_PATH, 'predictions_testing_%s_idx%d.bin'%(ID, halo_idx))

    halo_globals = eval(open(cfg._STORAGE_FILES['TNG']%(halo_idx, 'globals'), 'r').read())
    M200c = halo_globals['M200c']
    R200c = halo_globals['R200c']
    P200c = 100 * cfg._G_NEWTON * cfg._RHO_CRIT * cfg._OMEGA_B * M200c \
            / cfg._OMEGA_M / R200c

    box_target = np.fromfile(path_target, dtype=np.float32)
    N = cfg.TNG_RESOLUTION
    x = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2].transpose(1,2,3,0).reshape((N*N*N, 3)).astype(np.float32)
    x *= 5.0 / N # now in the range [-2.5, 2.5]
    r = np.linalg.norm(x, axis=-1).flatten()
    mask = r > cfg.RMAX
    box_target[mask] = 0
    box_target = box_target.reshape((N, N, N))

    # GNFW model
    A_P0 = 3.9183
    am_P0 = 0.5705
    A_xc = 2.8859
    am_xc = -0.8130
    A_beta = 13.8758
    am_beta = -0.6282

    P0 = A_P0 * (0.7 * M200c / 1e4)**am_P0
    xc = A_xc * (0.7 * M200c / 1e4)**am_xc
    beta = A_beta * (0.7 * M200c / 1e4)**am_beta

    # be careful to do the binning identically
    box_b12 = P0 * ((r+1e-3)/xc)**(-0.3) * (1 + (r+1e-3)/xc)**(-beta)
    box_b12[mask] = 0
    box_b12 = box_b12.reshape((N, N, N))

    box_prediction = CubifyPrediction(path_prediction)

    if cfg.SCALE_PTH :
        box_target /= P200c
    else :
        box_b12 *= P200c

    proj_ax = RNG.integers(3)

    img_target = np.nan_to_num(np.log(np.sum(box_target, axis=proj_ax)), posinf=np.NaN, neginf=np.NaN)
    img_b12 = np.nan_to_num(np.log(np.sum(box_b12, axis=proj_ax)), posinf=np.NaN, neginf=np.NaN)
    img_prediction = np.nan_to_num(np.log(np.sum(box_prediction, axis=proj_ax)), posinf=np.NaN, neginf=np.NaN)

    vmin = min((np.nanmin(img_target), np.nanmin(img_prediction), np.nanmin(img_b12)))
    vmax = min((np.nanmax(img_target), np.nanmax(img_prediction), np.nanmax(img_b12)))

    a[0].matshow(img_target, vmin=vmin, vmax=vmax, extent=(-2.5, 2.5, -2.5, 2.5))
    a[1].matshow(img_prediction, vmin=vmin, vmax=vmax, extent=(-2.5, 2.5, -2.5, 2.5))
    a[2].matshow(img_b12, vmin=vmin, vmax=vmax, extent=(-2.5, 2.5, -2.5, 2.5))

    exponent = int(np.log10(M200c))
    mantissa = M200c / 10**exponent
    a[0].set_title(r'$%.1f \times 10^{%d}\,h^{-1}M_\odot$'%(mantissa, 10+exponent), fontsize='x-small')

    for _a in a :
        _a.set_xlim(-2, 2)
        _a.set_ylim(-2, 2)

        _a.set_xticks([])
        _a.set_yticks([])

        for s in ['top', 'bottom', 'left', 'right', ] :
            _a.spines[s].set_visible(False)

ax[0][0].set_ylabel('target', fontsize='x-small')
ax[0][1].set_ylabel('reconstruction', fontsize='x-small')
ax[0][2].set_ylabel('benchmark', fontsize='x-small')


fig.savefig('images.pdf', bbox_inches='tight')
