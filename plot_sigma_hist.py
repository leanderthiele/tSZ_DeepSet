"""
command line arguments:
    [1...] ID(s)
"""

from sys import argv
from time import ctime
import os.path

import numpy as np
from matplotlib import pyplot as plt

import cfg

fig, ax = plt.subplots(ncols=2)

ax_lin = ax[0]
ax_log = ax[1]

sigma_bins = np.arange(-5, 5, dtype=float)
sigma_bins = np.concatenate((np.array([-np.inf, ]), sigma_bins, np.array([+np.inf, ])))

for _argv in argv[1:] :
    ID = _argv.replace('testing_', '').replace('.sbatch', '')

    fname = os.path.join(cfg.RESULTS_PATH, 'loss_testing_%s.npz'%ID)

    print('Using file %s'%fname)
    print('\tlast modified: %s'%ctime(os.path.getmtime(fname)))

    with np.load(fname) as f :
        idx_all = f['idx']
        recon_loss_all = f['loss']
        gauss_loss_all = f['gauss_loss']
        N_TNG_all = f['N_TNG']

    idx = np.unique(idx_all)
    recon_loss = np.empty(len(idx))
    gauss_loss = np.empty((len(idx), gauss_loss_all.shape[1]))
    for ii, this_idx in enumerate(idx) :
        mask = this_idx == idx_all
        recon_loss[ii] = np.sum(N_TNG_all[mask] * recon_loss_all[mask]) / np.sum(N_TNG_all[mask])
        gauss_loss[ii] = np.sum(N_TNG_all[mask][:, None] * gauss_loss_all[mask], axis=0) / np.sum(N_TNG_all[mask])

    color = None

    for ii, transf in enumerate([lambda x: x, lambda x: np.log(x), ]) :
        recon = transf(recon_loss)
        gauss = transf(gauss_loss)

        avg_gauss = np.mean(gauss, axis=1)
        std_gauss = np.std(gauss, axis=1)

        sigma = (recon - avg_gauss) / std_gauss

        hist, edges = np.histogram(sigma, bins=sigma_bins)

        if np.isinf(edges[0]) :
            edges[0] = edges[1] - 2
        if np.isinf(edges[-1]) :
            edges[-1] = edges[-2] + 2

        l = ax[ii].step(edges+3e-1*(np.random.rand()-0.5),
                        np.concatenate((hist, np.array([np.NaN, ]))),
                        where='post', label=ID)

        if color is None :
            color = plt.getp(l[0], 'color')


ax_lin.set_title('linear')
ax_log.set_title('logarithmic')

for a in ax :
    a.set_xlabel('(reconstruction-<gaussian>)/$\sigma$')
    a.set_ylabel('counts')

ax_lin.legend()

plt.show()
