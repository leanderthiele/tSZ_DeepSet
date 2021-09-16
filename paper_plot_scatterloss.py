import os.path
from time import ctime

import numpy as np
from matplotlib import pyplot as plt

import cfg

IDs = {'origin64': 'Origin+GNFW',
       'local64': 'Local',
       'localorigin64': 'Local+Origin+GNFW',
       'vae64_nr440': 'Local+Origin+GNFW+Stochasticity'}

MARKER_SCALE = 1.0

fig, ax = plt.subplots()

for ID, label in IDs.items() :
    
    fname = os.path.join(cfg.RESULTS_PATH, 'loss_testing_%s.npz'%ID)
    print('Using file %s'%fname)
    print('\tlast modified: %s\n'%ctime(os.path.getmtime(fname)))
    
    with np.load(fname) as f :
        
        loss_all = f['loss']
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
        logM[ii] = logM_all[mask][0]
        assert all(abs(logM[ii]/logm-1) < 1e-5 for logm in logM_all[mask])

    vmin = 8.518
    vmax = 11.534
    ax.scatter(guess_loss, loss, label=label,
               s=MARKER_SCALE*(3+20*(logM-vmin)/(vmax-vmin)))

    if gauss_loss is not None :
        ax.violinplot(gauss_loss, positions=guess_loss, showmeans=True,
                      label='%s gaussian codes'%label)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel('GNFW loss')
ax.set_ylabel('network loss')

min_lim = 0.9*min((ax.get_xlim()[0], ax.get_ylim()[0]))
max_lim = 1.1*max((ax.get_xlim()[1], ax.get_ylim()[1]))
ax.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='dashed', color='black')
ax.set_xlim(min_lim, max_lim)
ax.set_ylim(min_lim, max_lim)

ax.legend(loc='upper left', frameon=False)

plt.show()
