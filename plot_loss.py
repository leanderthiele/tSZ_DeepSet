"""
command line arguments :
    [1] ID
    [2] epoch
"""

from sys import argv
import os.path

import numpy as np
from matplotlib import pyplot as plt

import cfg

ID = argv[1].replace('training_', '').replace('.sbatch', '')

epoch = int(argv[2])

# for how many halos we want to insert the index
Nidx = 4

with np.load(os.path.join(cfg.RESULTS_PATH, 'loss_%s.npz'%ID)) as f :
    epochs = f['training'].shape[0]
    print('max epoch = %d'%(epochs-1))
    all_t = f['training']
    all_tg = f['training_guess']
    t = f['training'][epoch,:]
    tg = f['training_guess'][epoch,:]
    tkld = f['training_KLD'][epoch,:]
    tkldmean = np.mean(f['training_KLD'], axis=-1)
    tlogm = f['training_logM'][epoch,:]
    tidx = f['training_idx'][epoch,:]
    tmean = np.median(f['training'], axis=-1)
    tgmean = np.median(f['training_guess'], axis=-1)
    all_v = f['validation']
    all_vg = f['validation_guess']
    v = f['validation'][epoch,:]
    vg = f['validation_guess'][epoch,:]
    vkld = f['validation_KLD'][epoch,:]
    vlogm = f['validation_logM'][epoch,:]
    vidx = f['validation_idx'][epoch,:]
    vmean = np.median(f['validation'], axis=-1)
    vgmean = np.median(f['validation_guess'], axis=-1)
    vkldmean = np.mean(f['validation_KLD'], axis=-1)
    if 'validation_gauss' in f and len(f['validation_gauss']) > 0 :
        vgauss = f['validation_gauss'][epoch,:]
        all_vgauss = f['validation_gauss']
    else :
        vgauss = None
        all_vgauss = None

# over all masses, log(M200_CM)
vmin = 8.518
vmax = 11.534

fig, ax = plt.subplots(ncols=2, nrows=2)

ax_epoch_loss = ax[0][0]
ax_mean_loss = ax[0][1]
ax_epoch_kld = ax[1][0]
ax_mean_kld = ax[1][1]

ax_epoch_loss.scatter(tg, t, label='training', s=3+20*(tlogm-vmin)/(vmax-vmin))
ax_epoch_loss.scatter(vg, v, label='validation', s=3+20*(vlogm-vmin)/(vmax-vmin))
if vgauss is not None :
    ax_epoch_loss.scatter(vg, np.mean(vgauss, axis=-1), label='<validation gauss>', s=3+20*(vlogm-vmin)/(vmax-vmin))
    for ii in range(vgauss.shape[1]) :
        ax_epoch_loss.scatter(vg, vgauss[:, ii], label='validation gauss' if ii==0 else None, c='black', s=0.1)
ax_epoch_loss.set_yscale('log')
ax_epoch_loss.set_xscale('log')
min_lim = 0.9*min((ax_epoch_loss.get_xlim()[0], ax_epoch_loss.get_ylim()[0]))
max_lim = 1.1*max((ax_epoch_loss.get_xlim()[1], ax_epoch_loss.get_ylim()[1]))
ax_epoch_loss.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='dashed', color='black')
ax_epoch_loss.set_xlim(min_lim, max_lim)
ax_epoch_loss.set_ylim(min_lim, max_lim)

if Nidx :
    assert isinstance(Nidx, int) and Nidx > 0
    sorter = np.argsort(t)
    tg_sorted = tg[sorter][::-1]
    t_sorted = t[sorter][::-1]
    tidx_sorted = tidx[sorter][::-1]
    for ii in range(Nidx) :
        ax_epoch_loss.text(tg_sorted[ii], t_sorted[ii], '%d'%tidx_sorted[ii],
                           ha='center', va='top', transform=ax_epoch_loss.transData)

ax_epoch_loss.set_xlabel('B12 loss')
ax_epoch_loss.set_ylabel('network loss')
ax_epoch_loss.legend(loc='upper left', frameon=False, ncol=2)
ax_epoch_loss.text(0.95, 0.05, 'epoch %d'%(epoch), transform=ax_epoch_loss.transAxes, ha='right', va='bottom', color='green')

ax_epoch_kld.scatter(tg, tkld, label='training', s=3+20*(tlogm-vmin)/(vmax-vmin))
ax_epoch_kld.scatter(vg, vkld, label='validation', s=3+20*(vlogm-vmin)/(vmax-vmin))
ax_epoch_kld.set_xlim(1e-3, 1e0)
ax_epoch_kld.set_xscale('log')
ax_epoch_kld.set_xlabel('B12 loss')
ax_epoch_kld.set_ylabel('KLD')
ax_epoch_kld.legend(frameon=False)

e = np.arange(epochs)
#ax[1].plot(e, tmean/tgmean, label='training')
#ax[1].plot(e, vmean/vgmean, label='validation')
ax_mean_loss.plot(e, np.median(all_t/all_tg, axis=-1), label='training')
ax_mean_loss.plot(e, np.median(all_v/all_vg, axis=-1), label='validation')
if all_vgauss is not None :
    ax_mean_loss.plot(e, np.median(np.mean(all_vgauss, axis=-1)/all_vg, axis=-1), label='validation gauss')
ax_mean_loss.set_xlabel('epoch')
ax_mean_loss.set_ylabel('median(network/B12)')
ax_mean_loss.legend(loc='upper right', frameon=False)
ax_mean_loss.set_yscale('log')
ax_mean_loss.plot([epoch, epoch], [0,1], color='green', alpha=0.3, linewidth=1)
ax_mean_loss.set_xlim(0, epochs-1)
#ax_mean_loss.set_ylim(None, 1)

ax_mean_kld.plot(e, tkldmean, label='training')
ax_mean_kld.plot(e, vkldmean, label='validation')
ax_mean_kld.set_xlabel('epoch')
ax_mean_kld.set_ylabel('mean(KLD)')
ax_mean_kld.set_xlim(0, epochs-1)
ax_mean_kld.legend(frameon=False)

fig.suptitle(ID, fontsize=5)

plt.show()


