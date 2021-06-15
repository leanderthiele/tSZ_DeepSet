import os.path

import numpy as np

import torch
import torch.nn as nn

from torchsummary import summary

from mpi_env_types import MPIEnvTypes
from data_modes import DataModes
from data_loader import DataLoader
from training_loss import TrainingLoss
from data_batch import DataBatch
from network_origin import NetworkOrigin
from global_fields import GlobalFields
from basis import Basis
from init_proc import InitProc
import cfg

InitProc(0)

# if we are on the head node, restrict number of threads so we don't disturb other users
if cfg.MPI_ENV_TYPE is MPIEnvTypes.NOGPU :
    torch.set_num_threads(5)

model = NetworkOrigin()

# give a summary for debugging
summary(model, [(352,3), (len(GlobalFields),), (len(Basis),3)], device='cpu')

model = model.to_device()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = TrainingLoss()
training_loader = DataLoader(mode=DataModes.TRAINING, load_TNG=False)
validation_loader = DataLoader(mode=DataModes.VALIDATION, load_TNG=False)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                steps_per_epoch=len(training_loader),
                                                epochs=500)

training_loss_arr = []
validation_loss_arr = []

for epoch in range(500) :

    print('epoch %d'%epoch)

    print('TRAINING')
    model.train()

    this_training_loss = np.empty((len(training_loader), 2))

    for t, data in enumerate(training_loader) :

        assert isinstance(data, DataBatch)

        optimizer.zero_grad()

        data = data.to_device()

        prediction = model(data.DM_coords,
                           u=data.u if len(GlobalFields) != 0 else None,
                           basis=data.basis if len(Basis) != 0 else None)

        # remove the singleton feature channel dimension
        prediction = prediction.squeeze(dim=1)

        cm = data.CM_DM
        if cfg.NORMALIZE_COORDS :
            cm /= data.R200c.unsqueeze(-1)

        prediction += cm

        target = data.pos_TNG
        if cfg.NORMALIZE_COORDS :
            target /= data.R200c.unsqueeze(-1)

        loss = loss_fn(prediction, target)

        loss.backward()

        if cfg.GRADIENT_CLIP is not None :
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP)

        optimizer.step()

        scheduler.step()

        training_loss = loss.item()
        guess_loss = loss_fn(cm, target).item()
        print('training loss = %f vs guess = %f %s %s'%(training_loss, guess_loss, '***' if guess_loss>0.1 else '   ', '+++' if training_loss>0.1 else '   '))

        this_training_loss[t, 0] = training_loss
        this_training_loss[t, 1] = guess_loss

    training_loss_arr.append(this_training_loss)
    
    # FIXME
    # torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, 'origin.pt'))

    print('VALIDATION')
    model.eval()

    this_validation_loss = np.empty((len(validation_loader), 2))

    for t, data in enumerate(validation_loader) :
        
        data = data.to_device()

        prediction = model(data.DM_coords,
                           u=data.u if len(GlobalFields) != 0 else None,
                           basis=data.basis if len(Basis) != 0 else None)

        prediction = prediction.squeeze(dim=1)

        cm = data.CM_DM
        if cfg.NORMALIZE_COORDS :
            cm /= data.R200c.unsqueeze(-1)

        prediction += cm

        target = data.pos_TNG
        if cfg.NORMALIZE_COORDS :
            target /= data.R200c.unsqueeze(-1)

        loss = loss_fn(prediction, target)

        validation_loss = loss.item()
        guess_loss = loss_fn(cm, target).item()
        print('validation loss = %f vs guess = %f %s %s'%(validation_loss, guess_loss, '***' if guess_loss>0.1 else '   ', '+++' if validation_loss>0.1 else '   '))

        this_validation_loss[t, 0] = validation_loss
        this_validation_loss[t, 1] = guess_loss

    validation_loss_arr.append(this_validation_loss)

    np.savez(os.path.join(cfg.RESULTS_PATH, 'loss_origin.npz'),
             training=np.array(training_loss_arr),
             validation=np.array(validation_loss_arr))
