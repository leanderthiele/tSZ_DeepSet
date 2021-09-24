import os.path
from time import ctime

import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase

import cfg

IDs = {'origin64_ontestingset': ('Origin+GNFW', 'blue'),
       'local64_ontestingset': ('Local', 'cyan'),
       'localorigin64_again_200epochs_wbasis_nr116_ontestingset': ('Local+Origin+GNFW', 'magenta'),
       'vae64_200epochs_usekld_onelatent_nr2074_ontestingset': ('Local+Origin+GNFW+Stochasticity', 'green')}

MARKER_SCALE = 0.5

EXCESS_SPACE = 0.1

legend_objs = {}
legend_handlers = {}
splines = []
markers = []

fig, ax = plt.subplots(figsize=(7,5))
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
    markers.extend(zip(guess_loss, loss))
    legend_objs[ax.scatter(guess_loss, loss, s=MARKER_SCALE*(3+20*(logM-vmin)/(vmax-vmin)), c=color)] \
        = '%s %s ($\mathcal{L}_{\sf opt}=%.2f$)'%(label, '' if gauss_loss is None else 'reconstructed', loss_quantifier)
               

    spline_kwargs = {} # TODO deal with this later

    sorter = np.argsort(guess_loss)
    spl = UnivariateSpline(np.log(guess_loss[sorter]), np.log(loss[sorter]), **spline_kwargs)
    splines.append(spl)
    x = np.linspace(np.min(np.log(guess_loss)), np.max(np.log(guess_loss)))
    y = spl(x)
    ax.plot(np.exp(x), np.exp(y), color=color)

    if gauss_loss is not None :
        OPACITY = 0.5

        violinplot_kwargs = dict(showmeans=False, showextrema=False, widths=0.02)

        loss_quantifier = np.median(np.mean(gauss_loss, axis=-1)/guess_loss)

        lg = np.log(guess_loss)
        lg_min = np.min(lg)
        lg_max = np.max(lg)
        # plot this on the [0, 1] x-scale
        parts = ax_linear.violinplot(gauss_loss.T, positions=(lg-lg_min)/(lg_max-lg_min),
                                     **violinplot_kwargs)
        for pc in parts['bodies'] :
            pc.set_facecolor('none')
            pc.set_edgecolor(color)
            pc.set_alpha(OPACITY)

        # draw some fake stuff to attach label to
        fake_point = ax.scatter([0,], [0,], marker='')

        legend_objs[fake_point] = '%s sampled ($\mathcal{L}_{\sf opt}=%.2f$)'%(label, loss_quantifier)

        spl = UnivariateSpline(np.log(guess_loss[sorter]),
                               np.log(np.mean(gauss_loss, axis=-1)[sorter]), **spline_kwargs)
        splines.append(spl)
        y = spl(x)
        ax.plot(np.exp(x), np.exp(y), color=color, alpha=OPACITY)

        # let's do some *serious* hacking here to get a nice legend
        fig_fake, ax_fake = plt.subplots(figsize=(1,1), dpi=100)
        canvas = FigureCanvasAgg(fig_fake)
        parts_fake = ax_fake.violinplot(np.random.randn(1000), positions=[0,], **violinplot_kwargs)
        for pc in parts_fake['bodies'] :
            pc.set_facecolor('none')
            pc.set_edgecolor(color)
            pc.set_linewidth(5)
            pc.set_alpha(OPACITY)
        ax_fake.set_frame_on(False)
        ax_fake.set_xticks([])
        ax_fake.set_yticks([])
        ax_fake.set_xlim(-2*violinplot_kwargs['widths'], 2*violinplot_kwargs['widths'])
        fig_fake.tight_layout(pad=0)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        
        class ImgHandler(HandlerBase) :
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans) :
                sx, sy = self.img_stretch
                bb = Bbox.from_bounds(xdescent-sx, ydescent-sy, width+2*sx, height+2*sy)
                tbb = TransformedBbox(bb, trans)
                img = BboxImage(tbb)
                img.set_data(self.img_data)
                img.set_interpolation('hermite')
                self.update_prop(img, orig_handle, legend)
                return [img,]
            def set_img(self, a, img_stretch=(0,2)) :
                self.img_data = a
                self.img_stretch = img_stretch
        
        custom_handler = ImgHandler()
        custom_handler.set_img(buf)

        legend_handlers[fake_point] = custom_handler

ax.set_xlabel('benchmark loss')
ax.set_ylabel('network loss')

ax.set_yscale('log')
ax.set_xscale('log')

min_lim = (1-EXCESS_SPACE)*min((ax.get_xlim()[0], ax.get_ylim()[0]))
max_lim = (1+EXCESS_SPACE)*max((ax.get_xlim()[1], ax.get_ylim()[1]))
ax.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='dashed', color='grey', zorder=-10)
#ax.set_xlim(min_lim, max_lim)
#ax.set_ylim(min_lim, max_lim)
ax.set_xlim(4e-3, 2e-1)
ax.set_ylim(7e-4, 2e-1)

# for all annotations
annotation_kwargs = dict(# arrow properties
                         arrowprops={'arrowstyle': '-|>', 'fill': True, 'edgecolor': 'grey', 'facecolor': 'grey'},
                         # text properties
                         ha='left', va='top', color='grey', bbox={'visible': False, 'boxstyle': 'square, pad=0'})

# annotate smoothing splines
arrow_end_x = 4e-2
arrow_end_y = min(np.exp(s(np.log(arrow_end_x))) for s in splines)
ax.annotate('smoothing\nsplines', (arrow_end_x, arrow_end_y), xytext=(4e-2,5e-3),
            **annotation_kwargs)

# annotate dashed line
arrow_end_x = 1e-2/1.5
ax.annotate('benchmark', (arrow_end_x, arrow_end_x), xytext=(6e-3/1.4,2e-2/1.5),
            **annotation_kwargs)

# annotate markers
arrow_end_x_lims = [2e-2, 4e-2]
arrow_end_x, arrow_end_y = min(filter(lambda x: arrow_end_x_lims[0] < x[0] < arrow_end_x_lims[1], markers),
                               key=lambda x: x[1])
annotation_kwargs['arrowprops']['relpos'] = (0, 0.75)
ax.annotate('each marker = one cluster,\nsize ~ mass', (arrow_end_x, arrow_end_y), xytext=(3e-2,1.5e-3),
            **annotation_kwargs)


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

ax.legend(list(legend_objs.keys()), list(legend_objs.values()), handler_map=legend_handlers,
          loc='upper left', frameon=False)

fig.savefig('scatterloss.pdf', bbox_inches='tight', pad=0, dpi=4000)
