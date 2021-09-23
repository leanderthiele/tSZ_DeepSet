"""
command line arguments :
    [1] ID
"""

from sys import argv
from time import ctime
import os.path

import numpy as np
from matplotlib import pyplot as plt

import cfg

fig, ax = plt.subplots(ncols=2)

ax_loss = ax[0]
ax_KLD = ax[1]

ax_loss.set_yscale('log')
ax_loss.set_xscale('log')

min_lim = None
max_lim = None

for _argv in argv[1:] :
    ID = _argv.replace('testing_', '').replace('.sbatch', '')

    fname = os.path.join(cfg.RESULTS_PATH, 'loss_testing_%s.npz'%ID)

    print('Using file %s'%fname)
    print('\tlast modified: %s'%ctime(os.path.getmtime(fname)))

    with np.load(fname) as f :
        loss_all = f['loss']
        KLD_all = f['KLD']
        guess_loss_all = f['guess_loss']
        logM_all = f['logM']
        idx_all = f['idx']
        N_TNG_all = f['N_TNG']
        if 'gauss_loss' in f and len(f['gauss_loss']) > 0 :
            gauss_loss_all = f['gauss_loss']
        else :
            gauss_loss_all = None

    idx = np.unique(idx_all)
    loss = np.empty(len(idx))
    KLD = np.empty(len(idx))
    guess_loss = np.empty(len(idx))
    gauss_loss = np.empty((len(idx),gauss_loss_all.shape[1])) if gauss_loss_all is not None else None
    logM = np.empty(len(idx))
    for ii, this_idx in enumerate(idx) :
        mask = this_idx == idx_all 
        # loss is MSE (no sqrt!), so this makes sense
        loss[ii] = np.sum(N_TNG_all[mask] * loss_all[mask]) / np.sum(N_TNG_all[mask])
        guess_loss[ii] = np.sum(N_TNG_all[mask] * guess_loss_all[mask]) / np.sum(N_TNG_all[mask])
        if gauss_loss is not None :
            gauss_loss[ii] = np.sum(N_TNG_all[mask][:,None] * gauss_loss_all[mask], axis=0) / np.sum(N_TNG_all[mask])
        KLD[ii] = KLD_all[mask][0]
        if abs(KLD[ii] > 1e-7) : # avoid divide by zero
            assert all(abs(KLD[ii]/kld-1) < 1e-5 for kld in KLD_all[mask])
        logM[ii] = logM_all[mask][0]
        assert all(abs(logM[ii]/logm-1) < 1e-5 for logm in logM_all[mask])

    vmin = 8.518
    vmax = 11.534

    ax_loss.scatter(guess_loss, loss, label='reconstruction %s'%ID, s=3+20*(logM-vmin)/(vmax-vmin))
    if gauss_loss is not None :
        ax_loss.scatter(guess_loss, np.mean(gauss_loss, axis=1), label='<gaussian> %s'%ID, s=3+20*(logM-vmin)/(vmax-vmin))
        for ii in range(gauss_loss.shape[1]) :
            ax_loss.scatter(guess_loss, gauss_loss[:, ii], label='gaussian %s'%ID if ii==0 else None, c='black', s=0.1)

    this_min_lim = 0.9*min((ax_loss.get_xlim()[0], ax_loss.get_ylim()[0]))
    this_max_lim = 1.1*max((ax_loss.get_xlim()[1], ax_loss.get_ylim()[1]))

    if min_lim is None :
        min_lim = this_min_lim
    else :
        min_lim = min((this_min_lim, min_lim))

    if max_lim is None :
        max_lim = this_max_lim
    else :
        max_lim = max((this_max_lim, max_lim))

    loss_quantifier = np.median(loss/guess_loss)
    ax_loss.text(0.95, 0.05, '%.4f'%loss_quantifier, transform=ax_loss.transAxes, ha='right', va='bottom')

    if gauss_loss is not None :
        gauss_loss_quantifier = np.median(np.mean(gauss_loss, axis=1) / guess_loss)
        ax_loss.text(0.95, 0.1, '%.4f'%gauss_loss_quantifier, transform=ax_loss.transAxes, ha='right', va='bottom')

    ax_KLD.scatter(guess_loss, KLD, s=3+20*(logM-vmin)/(vmax-vmin))

ax_loss.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='dashed', color='black')
ax_loss.set_xlim(min_lim, max_lim)
ax_loss.set_ylim(min_lim, max_lim)
ax_loss.set_xlabel('B12 loss')
ax_loss.set_ylabel('network loss')
ax_loss.legend(loc='upper left', frameon=False)

ax_KLD.set_xscale('log')
ax_KLD.set_xlim(min_lim, max_lim)
ax_KLD.set_xlabel('B12 loss')
ax_KLD.set_ylabel('KLD')
ax_KLD.legend(frameon=False)

plt.show()
