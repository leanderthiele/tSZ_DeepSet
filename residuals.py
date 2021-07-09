# TODO ensure that we are noise-less, load network from file (must be consistent)

import os.path

import numpy as np

import torch
import torch.nn as nn

from network import Network
from network_batt12 import NetworkBatt12
from data_loader import DataLoader
from data_batch import DataBatch
from data_modes import DataModes
import cfg

InitProc(0)

model = Network().to_device()
checkpoint = torch.load(os.path.join(cfg.RESULTS_PATH, 'model_%s.pt'%cfg.NET_ID))
model.load_state_dict(checkpoint, strict=True)

model.eval()

loader = DataLoader(mode=DataModes.TRAINING)

for t, data in enumerate(loader) :

    assert isinstance(data, DataBatch)
    data = data.to_device()
    prediction = model(data)

    r_npy = data.TNG_radii.cpu().detach().numpy().squeeze()
    p_npy = prediction.cpu().detach().numpy().squeeze()
    t_npy = data.TNG_Pth.cpu().detach().numpy().squeeze()

    assert len(r_npy) == cfg.DATALOADER_ARGS['batch_size']
    assert len(r_npy) == len(p_npy)
    assert len(r_npy) == len(t_npy)

    for ii in range(cfg.DATALOADER_ARGS['batch_size']) :

        np.savez(os.path.join('/scratch/gpfs/lthiele/tSZ_DeepSet_pca', '%d.npz'%data.idx[ii]),
                 r=r, prediction=prediction, target=target)
