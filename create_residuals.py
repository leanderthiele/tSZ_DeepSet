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

loader = DataLoader(mode=DataModes.ALL, load_TNG_residuals=False)

RBINS = np.linspace(cfg.RESIDUALS_RMIN, cfg.RESIDUALS_RMAX, num=cfg.RESIDUALS_NBINS+1)
RCENTERS = 0.5*(RBINS[:-1] + RBINS[1:])

halo_catalog = dict(np.load(cfg.HALO_CATALOG))

# store intermediate data here
binned_residuals = np.zeros((halo_catalog['Nobjects'], cfg.RESIDUALS_NBINS))

def get_binned(x, indices) :
    out = np.empty(cfg.RESIDUALS_NBINS)
    for ii in range(cfg.RESIDUALS_NBINS) :
        out[ii] = np.mean(x[indices==ii])
    return out

assert cfg.TEST_ON_ALL

T = len(loader)

this_idx = -1
all_r = np.empty(0, dtype=np.float32)
all_p = np.empty(0, dtype=np.float32)
all_t = np.empty(0, dtype=np.float32)

for t, data in enumerate(loader) :

    assert isinstance(data, DataBatch)

    if data.idx[0] != this_idx and len(all_r) != 0 :
        # we have arrived at a new object and need to store the data for the old one
        sorter = np.argsort(all_r)
        all_r = all_r[sorter]
        all_p = all_p[sorter]
        all_t = all_t[sorter]

        indices = np.digitize(all_r, RBINS) - 1
        assert np.min(indices) == 0

        p_binned = get_binned(all_p, indices)
        t_binned = get_binned(all_t, indices)

        # safety measure
        assert np.count_nonzero(binned_residuals[this_idx, :]) == 0

        binned_residuals[this_idx, :] = (t_binned - p_binned) / p_binned   

        # reset arrays
        all_r = np.empty(0, dtype=np.float32)
        all_p = np.empty(0, dtype=np.float32)
        all_t = np.empty(0, dtype=np.float32)

    assert len(data) == 1
    this_idx = data.idx[0]
    print('%d / %d, idx = %d'%(t, T, this_idx))

    data = data.to_device()

    with torch.no_grad() :
        prediction, _ = model(data)

    all_p = np.concatenate((all_p, prediction.cpu().detach().numpy().squeeze()))
    all_t = np.concatenate((all_t, data.TNG_Pth.cpu().detach().numpy().squeeze()))
    all_r = np.concatenate((all_r, data.TNG_radii.cpu().detach().numpy().squeeze()))


# now we have all the binned data and should normalize it
# We only use the training set for normalization purposes
training_binned_residuals = binned_residuals[DataModes.TRAINING.sample_indices(len(binned_residuals))]

binned_residuals -= np.mean(training_binned_residuals, axis=0, keepdims=True)
binned_residuals /= np.std(training_binned_residuals, axis=0, keepdims=True)

# now save to files
for ii, x in enumerate(binned_residuals) :
    x.astype(np.float32).tofile(cfg._STORAGE_FILES['TNG']%(ii, 'residuals_%s'%cfg.RESIDUALS_NET_ID))
