"""
command line arguments :
    [1] file
    [2] epoch
"""

from sys import argv
import os.path

import numpy as np
from matplotlib import pyplot as plt

import cfg

fname = argv[1]
epoch = int(argv[2])

# for how many halos we want to insert the index
Nidx = 4

with np.load(os.path.join(cfg.RESULTS_PATH, fname)) as f :
    epochs = f['training'].shape[0]
    print('max epoch = %d'%(epochs-1))
    all_t = f['training']
    all_tg = f['training_guess']
    t = f['training'][epoch,:]
    tg = f['training_guess'][epoch,:]
    tlogm = f['training_logM'][epoch,:]
    tidx = f['training_idx'][epoch,:]
    tmean = np.median(f['training'], axis=-1)
    tgmean = np.median(f['training_guess'], axis=-1)
    all_v = f['validation']
    all_vg = f['validation_guess']
    v = f['validation'][epoch,:]
    vg = f['validation_guess'][epoch,:]
    vlogm = f['validation_logM'][epoch,:]
    vidx = f['validation_idx'][epoch,:]
    vmean = np.median(f['validation'], axis=-1)
    vgmean = np.median(f['validation_guess'], axis=-1)

# over all masses, log(M200_CM)
vmin = 8.518
vmax = 11.534

fig, ax = plt.subplots(ncols=2)

ax[0].scatter(tg, t, label='training', s=3+20*(tlogm-vmin)/(vmax-vmin))
ax[0].scatter(vg, v, label='validation', s=3+20*(vlogm-vmin)/(vmax-vmin))
ax[0].plot([0,1e3], [0,1e3], linestyle='dashed', color='black')
ax[0].set_xlim(1e-3,1e0)
ax[0].set_ylim(1e-3,1e0)
ax[0].set_yscale('log')
ax[0].set_xscale('log')

if Nidx :
    assert isinstance(Nidx, int) and Nidx > 0
    sorter = np.argsort(t)
    tg_sorted = tg[sorter][::-1]
    t_sorted = t[sorter][::-1]
    tidx_sorted = tidx[sorter][::-1]
    for ii in range(Nidx) :
        ax[0].text(tg_sorted[ii], t_sorted[ii], '%d'%tidx_sorted[ii],
                   ha='center', va='top', transform=ax[0].transData)

ax[0].set_xlabel('B12 loss')
ax[0].set_ylabel('network loss')
ax[0].legend(loc='upper left', frameon=False)
ax[0].text(0.95, 0.05, 'epoch %d'%(epoch), transform=ax[0].transAxes, ha='right', va='bottom', color='green')

e = np.arange(epochs)
#ax[1].plot(e, tmean/tgmean, label='training')
#ax[1].plot(e, vmean/vgmean, label='validation')
ax[1].plot(e, np.median(all_t/all_tg, axis=-1), label='training')
ax[1].plot(e, np.median(all_v/all_vg, axis=-1), label='validation')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('median(network/B12)')
ax[1].legend(loc='upper right', frameon=False)
ax[1].set_yscale('log')
ax[1].plot([epoch, epoch], [0,1], color='green', alpha=0.3, linewidth=1)
ax[1].set_xlim(0, epochs-1)
ax[1].set_ylim(None, 1)

plt.show()


