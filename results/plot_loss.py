"""
pass epoch as command line argument
"""

from sys import argv

import numpy as np
from matplotlib import pyplot as plt

epoch = int(argv[1])

with np.load('loss.npz') as f :
    epochs = f['training'].shape[0]
    print('max epoch = %d'%(epochs-1))
    t = f['training'][epoch,:]
    tg = f['training_guess'][epoch,:]
    tlogm = f['training_logM'][epoch,:]
    tmean = np.mean(f['training'], axis=-1)
    tgmean = np.mean(f['training_guess'], axis=-1)

# over all masses, log(M200_CM)
vmin = 8.518
vmax = 11.534

fig, ax = plt.subplots(ncols=2)

ax[0].scatter(tg, t, label='training', s=3+20*(tlogm-vmin)/(vmax-vmin))
ax[0].plot([0,3], [0,3], linestyle='dashed', color='black')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].legend(loc='upper left', frameon=False)
ax[0].text(0.95, 0.05, 'epoch %d'%(epoch), transform=ax[0].transAxes, ha='right', va='bottom', color='green')

e = np.arange(epochs)
ax[1].plot(e, tmean/tgmean, label='training')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('loss')
ax[1].legend(loc='upper right', frameon=False)
ax[1].set_yscale('log')
ax[1].plot([epoch, epoch], [0,1e2], color='green', alpha=0.3, linewidth=1)
ax[1].set_xlim(0, epochs-1)

plt.show()


