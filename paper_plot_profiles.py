import os.path
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from get_profiles import GetProfiles, RCENTERS, RMIN, RMAX, NR
from paper_plot_get_indices import GetIndices
import cfg

NCOLS = 2
NROWS = 2

# the network ID for which we want to plot
ID = 'vae64_200epochs_usekld_onelatent_nr2074_ontestingset'

halo_indices = GetIndices(NCOLS*NROWS)

fig, ax = plt.subplots(ncols=NCOLS, nrows=NROWS, figsize=(5,5),
                       gridspec_kw={'wspace': 0.3, 'hspace': 0.05})

if NCOLS * NROWS > 1 :
    ax = ax.flatten()
else :
    ax = [ax, ]

for a, halo_idx in zip(ax, halo_indices) :

    halo_globals = eval(open(cfg._STORAGE_FILES['TNG']%(halo_idx, 'globals'), 'r').read())
    M200c = halo_globals['M200c']

    target, prediction, gaussian_prediction, b12 = GetProfiles(ID, halo_idx)

    l = a.plot(RCENTERS, prediction, label='reconstruction')
    a.plot(RCENTERS, b12, label='GNFW')
    a.plot(RCENTERS, target, label='target', linestyle='none', marker='o', markersize=2)

    # confidence intervals
    upper = np.empty(NR)
    lower = np.empty(NR)
    for ii in range(NR) :
        samples = np.sort(np.array([gp[ii] for gp in gaussian_prediction]))
        upper[ii] = samples[-int(0.005*len(samples))]
        lower[ii] = samples[int(0.005*len(samples))]
    a.fill_between(RCENTERS, lower, upper, label='sampling 99% interval', alpha=0.3, color=plt.getp(l[0], 'color'))

    a.set_yscale('log')
    a.set_xlim(RMIN, RMAX)

    exponent = int(np.log10(M200c))
    mantissa = M200c / 10**exponent
    a.text(0.95, 0.95, r'$%.1f \times 10^{%d}\,h^{-1}M_\odot$'%(mantissa, 10+exponent),
           transform=a.transAxes, ha='right', va='top')

for ii in range(NROWS) :
    ax[ii*NCOLS].set_ylabel('$P_e/P_{200}$')

for ii in range(NCOLS) :
    ax[(NROWS-1)*NCOLS+ii].set_xlabel('$r/R_{200}$')

for ii in range((NROWS-1)*NCOLS) :  
    ax[ii].set_xticklabels([])
    

ax[0].legend(loc='lower left', bbox_to_anchor=(0,1), ncol=2)

fig.savefig('profiles.pdf', bbox_inches='tight')
