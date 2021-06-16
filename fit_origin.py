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
summary(model, [(137,3), (len(GlobalFields),), (len(Basis),3)], device='cpu')

model = model.to_device()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = TrainingLoss()
training_loader = DataLoader(mode=DataModes.TRAINING, load_TNG=False)
validation_loader = DataLoader(mode=DataModes.VALIDATION, load_TNG=False)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                steps_per_epoch=len(training_loader),
                                                epochs=400)

training_loss_arr = []
training_guess_loss_arr = []
training_logM_arr = []

validation_loss_arr = []
validation_guess_loss_arr = []
validation_logM_arr = []

for epoch in range(400) :

    print('epoch %d, lr = %f'%(epoch, optimizer.param_groups[0]['lr']))

    print('TRAINING')
    model.train()

    bs = cfg.DATALOADER_ARGS['batch_size']
    this_training_loss = []
    this_training_guess_loss = []
    this_trainig_logM = []

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

        # do this separately so we have it for each halo
        training_loss = torch.linalg.norm(prediction-target, dim=-1).cpu().detach().numpy()
        guess_loss = torch.linalg.norm(cm-target, dim=-1).cpu().detach().numpy()

        this_training_loss.extend(training_loss)
        this_training_guess_loss.extend(guess_loss)
        this_trainig_logM.extend(np.log(data.M200c.cpu().detach().numpy()))

    training_loss_arr.append(np.array(this_training_loss))
    training_guess_loss_arr.append(np.array(this_training_guess_loss))
    training_logM_arr.append(np.array(this_trainig_logM))
    
    # FIXME
    # torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, 'origin.pt'))

    print('VALIDATION')
    model.eval()

    this_validation_loss = []
    this_validation_guess_loss = []
    this_validation_logM = []

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

        validation_loss = torch.linalg.norm(prediction-target, dim=-1).cpu().detach().numpy()
        guess_loss = torch.linalg.norm(cm-target, dim=-1).cpu().detach().numpy()

        this_validation_loss.extend(validation_loss)
        this_validation_guess_loss.extend(guess_loss)
        this_validation_logM.extend(np.log(data.M200c.cpu().detach().numpy()))

    validation_loss_arr.append(np.array(this_validation_loss))
    validation_guess_loss_arr.append(np.array(this_validation_guess_loss))
    validation_logM_arr.append(np.array(this_validation_logM))

    np.savez(os.path.join(cfg.RESULTS_PATH, 'loss_origin.npz'),
             training=np.array(training_loss_arr),
             training_guess=np.array(training_guess_loss_arr),
             training_logM=np.array(training_logM_arr),
             validation=np.array(validation_loss_arr),
             validation_guess=np.array(validation_guess_loss_arr),
             validation_logM=np.array(validation_logM_arr))
