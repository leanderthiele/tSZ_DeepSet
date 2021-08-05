import numpy as np
from matplotlib import pyplot as plt

nets = ['encoder', 'local', 'vae', 'deformer', ]

scalars = dict()
desc = dict()

with np.load('normalization_data.npz') as f :
    for s in nets :
        scalars[s] = f['scalars_%s'%s]
        desc[s] = str(f['desc_%s'%s])

for s in nets :
    
    fig, ax = plt.subplots(nrows=9, ncols=5, figsize=(20,20))
    ax = ax.flatten()

    a = scalars[s]
    for ii in range(a.shape[-1]) :
        ax[ii].hist(a[:,ii], bins=20)
        ax[ii].text(0.05, 0.05, str(ii), transform=ax[ii].transAxes)
        avg = np.mean(a[:,ii])
        std = np.std(a[:,ii])
        ax[ii].text(0.05, 0.95, 'mu=%f s=%f'%(avg,std), transform=ax[ii].transAxes)
    
    fig.suptitle(desc[s])
    
    fig.savefig('normalization_hists_%s.pdf'%s, bbox_inches='tight')
