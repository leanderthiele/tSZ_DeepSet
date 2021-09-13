# TODO ensure that we are noise-less, load network from file (must be consistent)

import os.path

import numpy as np

import torch
import torch.nn as nn

from init_proc import InitProc
from network import Network
from network_batt12 import NetworkBatt12
from data_loader import DataLoader
from data_batch import DataBatch
from data_modes import DataModes
import cfg

InitProc(0)

model = Network().to_device()
checkpoint = torch.load(os.path.join(cfg.RESULTS_PATH, 'model_%s.pt'%cfg.RESIDUALS_NET_ID))
model.load_state_dict(checkpoint, strict=True)

model.eval()

loader = DataLoader(mode=DataModes.ALL)

RBINS = np.linspace(cfg.RESIDUALS_RMIN, cfg.RESIDUALS_RMAX, num=cfg.RESIDUALS_NBINS+1)
RCENTERS = 0.5*(RBINS[:-1] + RBINS[1:])

# store intermediate data here
binned_residuals = np.zeros((len(loader), cfg.RESIDUALS_NBINS))

def get_binned(x, indices) :
    out = np.empty(cfg.RESIDUALS_NBINS)
    for ii in range(cfg.RESIDUALS_NBINS) :
        out[ii] = np.mean(x[indices==ii])
    return out

for t, data in enumerate(loader) :

    assert isinstance(data, DataBatch)
    data = data.to_device()

    with torch.no_grad() :
        prediction = model(data)

    r_npy = data.TNG_radii.cpu().detach().numpy().squeeze()
    p_npy = prediction.cpu().detach().numpy().squeeze()
    t_npy = data.TNG_Pth.cpu().detach().numpy().squeeze()

    for ii in range(len(r_npy)) :
        r = r_npy[ii]
        p = p_npy[ii]
        t = t_npy[ii]

        sorter = np.argsort(r)
        r = r[sorter]
        p = p[sorter]
        t = t[sorter]

        indices = np.digitize(r, RBINS) - 1
        assert np.min(indices) == 0

        p_binned = get_binned(p, indices)
        t_binned = get_binned(t, indices)

        # safety measure
        assert np.count_nonzero(binned_residuals[data.idx[ii], :]) == 0

        binned_residuals[data.idx[ii], :] = (t_binned - p_binned) / p_binned   

# now we have all the binned data and should normalize it
# We only use the training set for normalization purposes
training_binned_residuals = binned_residuals[DataModes.TRAINING.sample_indices(len(binned_residuals))]

binned_residuals -= np.mean(training_binned_residuals, axis=0, keepdims=True)
binned_residuals /= np.std(training_binned_residuals, axis=0, keepdims=True)

# now save to files
for ii, x in enumerate(binned_residuals) :
    x.astype(np.float32).tofile(cfg._STORAGE_FILES['TNG']%(ii, 'residuals_%s'%cfg.RESIDUALS_NET_ID))
