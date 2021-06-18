import os.path

import numpy as np

import torch
import torch.nn as nn

from mpi_env_types import MPIEnvTypes
from network import Network
from network_batt12 import NetworkBatt12
from data_loader import DataLoader
from data_modes import DataModes
from training_loss import TrainingLoss
from data_batch import DataBatch
from init_proc import InitProc
import cfg

EPOCHS = 500

InitProc(0)

if cfg.MPI_ENV_TYPE is MPIEnvTypes.NOGPU :
    torch.set_num_threads(5)

model = Network().to_device()

batt12 = NetworkBatt12().to_device() # use this to compute the reference loss
batt12.eval()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = TrainingLoss()

training_loader = DataLoader(mode=DataModes.TRAINING)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                steps_per_epoch=len(training_loader),
                                                epochs=EPOCHS)

# store all the losses here
training_loss_arr = []
training_guess_loss_arr = []


for epoch in range(EPOCHS) :

    model.train()
    print('epoch %d'%epoch)

    this_training_loss_arr = []
    this_training_guess_loss_arr = []

    for t, data in enumerate(training_loader) :

        assert isinstance(data, DataBatch)

        optimizer.zero_grad()
        data = data.to_device()
        prediction = model(data)

        with torch.no_grad() :
            guess = batt12(data.M200c, data.TNG_radii, R200c=data.R200c if not cfg.NORMALIZE_COORDS else None)

        loss, loss_list = loss_fn(prediction, data.TNG_Pth, w=data.P200c)
        _, loss_list_guess = loss_fn(guess, data.TNG_Pth, w=data.P200c)

        this_training_loss_arr.extend([l.item() for l in loss_list])
        this_training_guess_loss_arr.extend([l.item() for l in loss_list_guess])

        loss.backward()
        
#        if cfg.GRADIENT_CLIP is not None :
#            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP)

        optimizer.step()
        scheduler.step()


    # put the losses for this epoch in the global arrays
    training_loss_arr.append(this_training_loss_arr)
    training_guess_loss_arr.append(this_training_guess_loss_arr)


    # save all the losses so far to file 
    np.savez(os.path.join(cfg.RESULTS_PATH, 'loss.npz'),
             training=np.array(training_loss_arr),
             training_guess=np.array(training_guess_loss_arr))
