import os.path
from time import ctime

import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt

import cfg

IDs = {'origin64': ('Origin+GNFW', 'blue'),
       'local64': ('Local', 'green'),
       'localorigin64': ('Local+Origin+GNFW', 'magenta'),
       'vae64_nr440': ('Local+Origin+GNFW+Stochasticity', 'cyan')}

MARKER_SCALE = 0.5

EXCESS_SPACE = 0.1

fig, ax = plt.subplots(figsize=(5,5))
ax_linear = ax.twiny()

guess_loss = None

for ID, (label, color) in list(IDs.items())[::-1] :
    
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

    idx = np.sort(np.unique(idx_all))
    loss = np.empty(len(idx))
    guess_loss_here = np.zeros(len(idx))
    gauss_loss = np.empty((len(idx), gauss_loss_all.shape[1])) if gauss_loss_all is not None else None
    logM = np.empty(len(idx))

    for ii, this_idx in enumerate(idx) :
        mask = this_idx == idx_all 
        # loss is MSE (no sqrt!), so this makes sense
        loss[ii] = np.sum(N_TNG_all[mask] * loss_all[mask]) / np.sum(N_TNG_all[mask])
        guess_loss_here[ii] = np.sum(N_TNG_all[mask] * guess_loss_all[mask]) / np.sum(N_TNG_all[mask])
        if gauss_loss is not None :
            gauss_loss[ii] = np.sum(N_TNG_all[mask][:,None] * gauss_loss_all[mask], axis=0) / np.sum(N_TNG_all[mask])
        logM[ii] = logM_all[mask][0]
        assert all(abs(logM[ii]/logm-1) < 1e-5 for logm in logM_all[mask])

    if guess_loss is None :
        guess_loss = guess_loss_here
    else :
        assert np.allclose(guess_loss, guess_loss_here)

    loss_quantifier = np.median(loss/guess_loss)

    vmin = 8.518
    vmax = 11.534
    ax.scatter(guess_loss, loss, label='%s (%.2f)'%(label, loss_quantifier),
               s=MARKER_SCALE*(3+20*(logM-vmin)/(vmax-vmin)), c=color)

    spline_kwargs = {} # TODO deal with this later

    sorter = np.argsort(guess_loss)
    spl = UnivariateSpline(np.log(guess_loss[sorter]), np.log(loss[sorter]), **spline_kwargs)
    x = np.linspace(np.min(np.log(guess_loss)), np.max(np.log(guess_loss)))
    y = spl(x)
    ax.plot(np.exp(x), np.exp(y), color=color)

    if gauss_loss is not None :
        lg = np.log(guess_loss)
        lg_min = np.min(lg)
        lg_max = np.max(lg)
        # plot this on the [0, 1] x-scale
        parts = ax_linear.violinplot(gauss_loss.T, positions=(lg-lg_min)/(lg_max-lg_min),
                                     showmeans=False, showextrema=False, widths=0.02)
        for pc in parts['bodies'] :
            pc.set_facecolor('none')
            pc.set_edgecolor(color)

        spl = UnivariateSpline(np.log(guess_loss[sorter]),
                               np.log(np.mean(gauss_loss, axis=-1)[sorter]), **spline_kwargs)
        y = spl(x)
        ax.plot(np.exp(x), np.exp(y), color=color, linestyle='dashed')

ax.set_xlabel('GNFW loss')
ax.set_ylabel('network loss')

ax.set_yscale('log')
ax.set_xscale('log')

min_lim = (1-EXCESS_SPACE)*min((ax.get_xlim()[0], ax.get_ylim()[0]))
max_lim = (1+EXCESS_SPACE)*max((ax.get_xlim()[1], ax.get_ylim()[1]))
ax.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='dashed', color='black')
#ax.set_xlim(min_lim, max_lim)
#ax.set_ylim(min_lim, max_lim)
ax.set_xlim(3e-3, 2e-1)
ax.set_ylim(7e-4, 2e-1)

# now we need to figure out how to adjust limits on our fake axis with linear scale
lg = np.log(guess_loss)
lg_min = np.min(lg)
lg_max = np.max(lg)
l_min = np.log(ax.get_xlim()[0])
l_max = np.log(ax.get_xlim()[1])
add_max = (l_max - lg_max) / (lg_max - lg_min)
add_min = (l_min - lg_min) / (lg_max - lg_min)
ax_linear.set_xlim(add_min, 1+add_max)

ax_linear.set_xticks([])

ax.legend(loc='upper left', frameon=False)

fig.savefig('scatterloss.pdf', bbox_inches='tight')
